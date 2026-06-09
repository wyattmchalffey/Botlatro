"""A/B the clear-capacity model as a blended shop leaf, on winrate.

Deploys via SolverPolicy.shop_leaf_value_fn: leaf = heuristic_shop_leaf + W * SCALE * clearcap,
where clearcap = sigmoid(model.forward(leaf)) in [0,1] and SCALE is the heuristic leaf's std on
calibration shop states (so W~1 makes the learned term ~1 std). Keeps the tuned economy/slot/build
terms and adds the learned build-strength signal. Compares winrate + mean ante vs the heuristic.

    PYTHONPATH=src py -3.12 scripts/phase8_clearcap_ab.py --ckpt .data/clearcap_attn_v2.pt \
        --seeds 150 --jobs 8 --w 0.5 --w 1.0 --out .data/clearcap_ab.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_MODEL = None
_CKPT = None


def _clearcap(model, state):
    import torch
    from balatro_ai.ml.encoding import encode_state
    from balatro_ai.ml.model import collate_states
    with torch.no_grad():
        return float(torch.sigmoid(model.forward(collate_states([encode_state(state)])))[0])


def _mk_sim(seed):
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = SeedGame(seed, stake="white").initial_state()
    return sim


def _calib_scale(depth, cap):
    from balatro_ai.api.state import GamePhase
    from balatro_ai.search.shop_search import shop_leaf_value
    from balatro_ai.solver.policy import SolverPolicy, _has_shop_action
    driver = SolverPolicy(play_backend="v2", play_depth=depth, play_width=1, seed=0)
    vals = []
    for i in range(1, 6):
        sim = _mk_sim(f"{900000 + i:07d}")
        for _ in range(2000):
            st = sim.state
            if st.run_over or st.phase == GamePhase.RUN_OVER:
                break
            if st.phase == GamePhase.SHOP and _has_shop_action(st):
                vals.append(shop_leaf_value(st))
                if len(vals) >= cap:
                    break
            sim.step(driver.choose_action(st))
        if len(vals) >= cap:
            break
    return statistics.pstdev(vals) if len(vals) > 1 else 1.0


def _run_seed(arg):
    global _MODEL, _CKPT
    seed, condition, ckpt, scale, w, depth, width = arg
    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import GamePhase
    from balatro_ai.search.shop_search import shop_leaf_value
    from balatro_ai.solver.policy import SolverPolicy

    leaf_fn = None
    if condition != "heuristic":
        if _MODEL is None or _CKPT != ckpt:
            from balatro_ai.ml.train import load_checkpoint
            _MODEL = load_checkpoint(ckpt); _MODEL.eval(); _CKPT = ckpt
        model = _MODEL

        def factory(root_state):
            def value(leaf):
                return shop_leaf_value(leaf, root_state=root_state) + w * scale * _clearcap(model, leaf)
            return value
        leaf_fn = factory

    policy = SolverPolicy(play_backend="v2", play_depth=depth, play_width=width,
                          seed=0, shop_leaf_value_fn=leaf_fn)
    sim = _mk_sim(seed)
    for _ in range(2000):
        st = sim.state
        if st.run_over or st.phase == GamePhase.RUN_OVER:
            break
        a = policy.choose_action(st)
        if a.action_type == ActionType.NO_OP:
            break
        sim.step(a)
    s = sim.state
    return {"seed": seed, "condition": condition, "won": bool(s.won), "ante": s.ante}


def _agg(rows, name):
    antes = [r["ante"] for r in rows]
    wins = sum(int(r["won"]) for r in rows)
    return {"condition": name, "n": len(rows), "wins": wins,
            "winrate": round(wins / max(1, len(rows)), 3),
            "mean_ante": round(statistics.mean(antes), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", type=int, default=150)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--w", type=float, action="append", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ws = args.w or [0.5, 1.0]

    scale = _calib_scale(args.depth, 80)
    print(f"[clearcap-ab] heuristic-leaf std (scale) = {scale:.1f}; testing W={ws}", flush=True)
    seeds = [f"{i:07d}" for i in range(1, args.seeds + 1)]
    conds = ["heuristic"] + [f"blend_w{w}" for w in ws]
    wmap = {f"blend_w{w}": w for w in ws}
    jobs = []
    for cond in conds:
        w = wmap.get(cond, 0.0)
        jobs += [(s, cond, args.ckpt, scale, w, args.depth, args.width) for s in seeds]

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = list(ex.map(_run_seed, jobs))
    else:
        rows = [_run_seed(j) for j in jobs]

    results = [_agg([r for r in rows if r["condition"] == c], c) for c in conds]
    for r in results:
        print("[clearcap-ab]", json.dumps(r), flush=True)
    json.dump({"scale": scale, "ws": ws, "results": results}, open(args.out, "w", encoding="utf-8"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
