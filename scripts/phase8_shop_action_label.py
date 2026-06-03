"""Option C / shop action-ranker — PROTOTYPE step 1: validate the CRN labeler.

The value-as-V(state) approach failed because shop value lives in a tiny, action-
dependent residual that V can't resolve. The fix is to label ACTION-relative
counterfactual value: at a shop state, enumerate legal actions, apply each with the
forward model, and evaluate every branch with COMMON-RANDOM-NUMBER rollouts (same
seeds across actions, so the per-action delta cancels the brutal shared variance).

Before training a ranker, this script answers the GATE question: is that signal
(a) RELIABLE — does the action ranking reproduce across an independent half of the
    rollouts? (split-half), and
(b) does it have HEADROOM — does the labeler reliably prefer a DIFFERENT action than
    the heuristic shop on some states? (that's the edge a ranker could capture).
If reliability is low, no ranker can learn it; if headroom is ~0, there's nothing to
beat. Either way we learn it cheaply, before the expensive ranker build.

    PYTHONPATH=src python scripts/phase8_shop_action_label.py \
        --states 20 --rollouts 6 --max-antes 5 --jobs 8 \
        --metrics .data/phase8_shop_action_label.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _pearson(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx <= 0 or sy <= 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / ((sx * sy) ** 0.5)


def _frac(state) -> float:
    if state.required_score and state.required_score > 0:
        return min(1.0, max(0.0, state.current_score / state.required_score))
    return 0.0


_VMODEL = {}


def _get_value_model(path):
    if path not in _VMODEL:
        from balatro_ai.ml.train import load_checkpoint
        m = load_checkpoint(path)
        m.eval()
        _VMODEL[path] = m
    return _VMODEL[path]


def _truncation_value(state, ckpt_path) -> float:
    """Coarse value-at-truncation: the value head's predicted FINAL ante from the
    bounded-rollout cutoff — captures build payoff that lands beyond the horizon."""
    import torch
    from balatro_ai.ml.encoding import encode_state
    from balatro_ai.ml.model import collate_states
    model = _get_value_model(ckpt_path)
    with torch.no_grad():
        return float(model.ante_value(collate_states([encode_state(state)]))[0]) * 9.0


def _action_key(a):
    return (str(a.action_type), a.target_id, a.amount)


def _shop_actions(state):
    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import with_derived_legal_actions
    shop_types = {ActionType.BUY, ActionType.SELL, ActionType.REROLL, ActionType.END_SHOP}
    s = with_derived_legal_actions(state)
    acts = [a for a in s.legal_actions if a.action_type in shop_types]
    # Keep all buys/reroll/end; cap sells (the failure mode is over-selling, 2 is enough),
    # and cap total to bound rollout cost.
    non_sell = [a for a in acts if a.action_type != ActionType.SELL]
    sell = [a for a in acts if a.action_type == ActionType.SELL][:2]
    return (non_sell + sell)[:10]


def _action_value(state, action, seed, rollout_bot, max_antes, value_ckpt=None, max_steps=300):
    """Apply `action` to `state`, then roll out (bounded) under the rollout policy.
    The sim is seeded by `seed` → using the SAME seed across actions = CRN pairing.
    At the horizon cap (still alive), use the value head's final-ante estimate if
    `value_ckpt` is set (captures delayed build payoff); else use ante+frac so far."""
    from dataclasses import replace as _replace

    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import GamePhase
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    sim = LocalBalatroSimulator(seed=seed, stake="white")
    sim.state = state
    try:
        sim.step(action)
    except (ValueError, IndexError, KeyError, TypeError, AttributeError):
        return None
    bot = create_bot(rollout_bot, seed=seed)
    start = state.ante
    with bot_config_scope(_replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        for _ in range(max_steps):
            s = sim.state
            if s.won:
                return 9.0
            if s.run_over or s.phase == GamePhase.RUN_OVER:
                break
            if s.ante - start >= max_antes:
                if value_ckpt:
                    return _truncation_value(s, value_ckpt)
                return float(s.ante) + _frac(s)
            a = bot.choose_action(s)
            if a is None or a.action_type == ActionType.NO_OP:
                break
            try:
                sim.step(a)
            except (ValueError, IndexError, KeyError, TypeError, AttributeError):
                break
    f = sim.state
    return float(f.ante) + _frac(f)


def _label_one_state(arg):
    state, seeds, rollout_bot, max_antes, value_ckpt = arg
    from balatro_ai.bots.registry import create_bot

    actions = _shop_actions(state)
    if len(actions) < 2:
        return None
    # heuristic shop pick (the baseline to beat)
    try:
        hp_key = _action_key(create_bot(rollout_bot, seed=0).choose_action(state))
    except Exception:
        hp_key = None

    vals = {}
    for a in actions:
        vs, ok = [], True
        for sd in seeds:
            v = _action_value(state, a, sd, rollout_bot, max_antes, value_ckpt)
            if v is None:
                ok = False
                break
            vs.append(v)
        if ok:
            vals[_action_key(a)] = vs
    if len(vals) < 2:
        return None

    keys = list(vals.keys())
    half = len(seeds) // 2
    meanv = {k: statistics.mean(vals[k]) for k in keys}
    a_half = {k: statistics.mean(vals[k][:half]) for k in keys}
    b_half = {k: statistics.mean(vals[k][half:]) for k in keys}
    bestA = max(keys, key=lambda k: a_half[k])
    bestB = max(keys, key=lambda k: b_half[k])
    best_full = max(keys, key=lambda k: meanv[k])
    half_corr = _pearson([a_half[k] for k in keys], [b_half[k] for k in keys])
    hp_in = hp_key in meanv
    headroom = meanv[best_full] - (meanv[hp_key] if hp_in else min(meanv.values()))
    return {
        "n_actions": len(keys),
        "top1_stable": 1.0 if bestA == bestB else 0.0,
        "half_corr": half_corr,
        "agree_heuristic": (1.0 if best_full == hp_key else 0.0) if hp_in else None,
        "headroom_vs_heuristic": headroom if hp_in else None,
        # labeler reliably prefers a NON-heuristic action (real, reproducible edge):
        "reliable_disagree": 1.0 if (hp_in and best_full != hp_key
                                     and bestA == bestB and headroom > 0.05) else 0.0,
        "value_spread": statistics.pstdev(list(meanv.values())) if len(keys) > 1 else 0.0,
    }


def _collect_shop_states(seeds, cap, per_seed, max_antes_skip=0):
    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import GamePhase
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from dataclasses import replace as _replace

    shop_types = {ActionType.BUY, ActionType.REROLL, ActionType.END_SHOP}
    out = []
    with bot_config_scope(_replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        for sd in seeds:
            sim = LocalBalatroSimulator(seed=_stable_seed_int(sd), stake="white")
            sim.state = SeedGame(sd, stake="white").initial_state()
            bot = create_bot("basic_strategy_bot", seed=0)
            taken = 0
            for _ in range(2000):
                st = sim.state
                if st.run_over or st.phase == GamePhase.RUN_OVER:
                    break
                if st.phase == GamePhase.SHOP:
                    shop_acts = [a for a in st.legal_actions if a.action_type in shop_types]
                    # require a real choice: at least one BUY + the option to skip
                    if sum(1 for a in shop_acts if a.action_type == ActionType.BUY) >= 1 and len(shop_acts) >= 2:
                        out.append(st)
                        taken += 1
                        if len(out) >= cap:
                            return out
                        if taken >= per_seed:
                            break
                a = bot.choose_action(st)
                if a is None:
                    break
                sim.step(a)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=20)
    ap.add_argument("--rollouts", type=int, default=16, help="CRN seeds per action (even, for split-half)")
    ap.add_argument("--max-antes", type=int, default=6, help="bounded rollout horizon")
    ap.add_argument("--per-seed", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--rollout-bot", default="basic_strategy_bot")
    ap.add_argument("--value-ckpt", default="", help="coarse value head for truncation (empty=ante+frac)")
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    seeds = [f"{700000 + i:07d}" for i in range(1, 400)]
    states = _collect_shop_states(seeds, cap=args.states, per_seed=args.per_seed)
    print(f"[shop-label] collected {len(states)} shop states; "
          f"labeling each action with {args.rollouts} CRN rollouts (+{args.max_antes} ante)...", flush=True)

    crn = list(range(1, args.rollouts + 1))
    jobs = [(st, crn, args.rollout_bot, args.max_antes, args.value_ckpt or None) for st in states]
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = [r for r in ex.map(_label_one_state, jobs) if r is not None]
    else:
        results = [r for r in (_label_one_state(j) for j in jobs) if r is not None]

    if not results:
        print("[shop-label] no valid states", flush=True)
        return 1

    def _mean(key, filt=lambda r: True):
        xs = [r[key] for r in results if filt(r) and r[key] is not None]
        return round(statistics.mean(xs), 4) if xs else None

    with_heur = [r for r in results if r["agree_heuristic"] is not None]
    out = {
        "n_states": len(results),
        "rollouts_per_action": args.rollouts,
        "max_antes": args.max_antes,
        "value_truncation_ckpt": args.value_ckpt or None,
        "mean_actions_per_state": _mean("n_actions"),
        # RELIABILITY: does the ranking reproduce on an independent half?
        "top1_stable": _mean("top1_stable"),
        "mean_half_corr": _mean("half_corr"),
        "mean_value_spread": _mean("value_spread"),
        # HEADROOM: does the labeler reliably disagree with (beat) the heuristic?
        "agree_heuristic_rate": (round(statistics.mean([r["agree_heuristic"] for r in with_heur]), 4)
                                 if with_heur else None),
        "reliable_disagree_rate": _mean("reliable_disagree"),
        "mean_headroom_vs_heuristic": _mean("headroom_vs_heuristic"),
    }
    print("[shop-label] RESULT:", json.dumps(out, indent=2), flush=True)
    with open(args.metrics, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
