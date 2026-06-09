"""Option A, Part 1 (the decisive cheap sub-test): is the joker-value signal in
the TARGET, or only missing from the net?

The value head failed the joker-removal test (removing a joker didn't lower its
estimate). Two causes: (noise) single-trajectory outcome labels are too noisy for
the net to extract the signal, or (bias) under the rollout policy jokers genuinely
don't change the outcome. This tests the TARGET directly — no training:

For ante>=N states (where jokers actually decide outcomes), compute a PAIRED
rollout value with vs without each joker (same rollout seeds + re-derived legal
actions, so the joker is the only difference), bounded to +max-antes. If the
rollout-averaged value DROPS when a joker is removed, the signal exists in the
data => noise => relabel+retrain works. If flat, it's bias => need a stronger
rollout policy. Parallel over states. Compares against the NET's (flat) delta.

    PYTHONPATH=src python scripts/phase8_rollout_value_diagnostic.py \
        --states 12 --samples 5 --max-antes 2 --jobs 8 \
        --ckpt .data/phase8_value_v3_bootstrap.pt \
        --metrics .data/phase8_rollout_value_diag.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _frac(state) -> float:
    if state.required_score and state.required_score > 0:
        return min(1.0, max(0.0, state.current_score / state.required_score))
    return 0.0


def _rollout_value(state, *, seed, rollout_bot, max_antes, max_steps=300):
    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import GamePhase
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    sim = LocalBalatroSimulator(seed=seed, stake="white")
    sim.state = state
    bot = create_bot(rollout_bot, seed=seed)
    start_ante = state.ante
    for _ in range(max_steps):
        s = sim.state
        if s.won:
            return float(start_ante + max_antes) + 1.0  # capped success dominates
        if s.run_over or s.phase == GamePhase.RUN_OVER:
            break
        if s.ante - start_ante >= max_antes:
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


def _collect_states(seeds, min_jokers, cap, min_ante):
    from balatro_ai.api.state import GamePhase
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    drv = SolverPolicy(play_backend="v2", play_depth=2, play_width=1, seed=0)
    out = []
    for seed in seeds:
        sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
        sim.state = SeedGame(seed, stake="white").initial_state()
        for _ in range(1500):
            st = sim.state
            if st.run_over or st.phase == GamePhase.RUN_OVER:
                break
            if (st.ante >= min_ante and len(st.jokers) >= min_jokers
                    and st.phase in (GamePhase.SELECTING_HAND, GamePhase.SHOP)):
                out.append(st)
                if len(out) >= cap:
                    return out
            sim.step(drv.choose_action(st))
    return out


def _state_job(arg):
    """Per-state paired joker-removal deltas (parallel unit)."""
    st, si, samples, rollout_bot, max_antes = arg
    from balatro_ai.api.state import with_derived_legal_actions

    st_d = with_derived_legal_actions(st)
    njok = len(st.jokers)
    sums = [0.0] * njok
    base_vals = []
    for s in range(samples):
        seed = 7919 * (s + 1) + 31 * si
        v_with = _rollout_value(st_d, seed=seed, rollout_bot=rollout_bot, max_antes=max_antes)
        base_vals.append(v_with)
        for i in range(njok):
            without = with_derived_legal_actions(dataclasses.replace(
                st, jokers=tuple(j for k, j in enumerate(st.jokers) if k != i)))
            v_without = _rollout_value(without, seed=seed, rollout_bot=rollout_bot, max_antes=max_antes)
            sums[i] += (v_with - v_without)
    return [d / samples for d in sums], base_vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=12)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--max-antes", type=int, default=2)
    ap.add_argument("--min-ante", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--rollout-bot", default="basic_strategy_bot")
    ap.add_argument("--ckpt", default=".data/phase8_value_v3_bootstrap.pt")
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    seeds = [f"{700000 + i:07d}" for i in range(1, 41)]
    states = _collect_states(seeds, min_jokers=2, cap=args.states, min_ante=args.min_ante)
    print(f"[diag] {len(states)} states (ante>={args.min_ante}), "
          f"{sum(len(s.jokers) for s in states)} jokers; rollout +{args.max_antes} antes", flush=True)
    if not states:
        print("[diag] no qualifying states")
        return 1

    jobs = [(st, si, args.samples, args.rollout_bot, args.max_antes)
            for si, st in enumerate(states)]
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(_state_job, jobs))
    else:
        results = [_state_job(j) for j in jobs]

    roll_deltas = [d for r in results for d in r[0]]
    base_vals = [b for r in results for b in r[1]]

    # NET joker-removal delta on the SAME states (the known-flat baseline).
    import torch
    from balatro_ai.api.state import with_derived_legal_actions
    from balatro_ai.ml.encoding import encode_state
    from balatro_ai.ml.model import collate_states
    from balatro_ai.ml.train import load_checkpoint

    model = load_checkpoint(args.ckpt)
    model.eval()
    net_deltas = []
    with torch.no_grad():
        for st in states:
            base = float(model.ante_value(collate_states([encode_state(with_derived_legal_actions(st))]))[0])
            for i in range(len(st.jokers)):
                without = with_derived_legal_actions(dataclasses.replace(
                    st, jokers=tuple(j for k, j in enumerate(st.jokers) if k != i)))
                v = float(model.ante_value(collate_states([encode_state(without)]))[0])
                net_deltas.append(base - v)

    out = {
        "states": len(states), "samples": args.samples, "max_antes": args.max_antes,
        "rollout_bot": args.rollout_bot,
        "rollout_target": {
            "joker_removal_delta_mean": round(statistics.mean(roll_deltas), 4),
            "joker_removal_delta_std": round(statistics.pstdev(roll_deltas), 4),
            "frac_positive": round(sum(1 for d in roll_deltas if d > 0) / len(roll_deltas), 3),
            "n_jokers": len(roll_deltas),
            "base_value_mean": round(statistics.mean(base_vals), 3),
            "base_value_std": round(statistics.pstdev(base_vals), 3),
        },
        "net_ante_head_singletraj": {
            "joker_removal_delta_mean": round(statistics.mean(net_deltas), 4),
            "frac_positive": round(sum(1 for d in net_deltas if d > 0) / len(net_deltas), 3),
            "n": len(net_deltas),
        },
    }
    with open(args.metrics, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("[diag] RESULT:", json.dumps(out, indent=2), flush=True)
    rt = out["rollout_target"]
    open(f".data/_ROLLVALDIAG_targetdelta{rt['joker_removal_delta_mean']:.3f}"
         f"_targetfrac{rt['frac_positive']:.2f}", "w").close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
