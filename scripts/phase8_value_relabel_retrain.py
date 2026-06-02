"""Option A Part 2: relabel states with multi-rollout-AVERAGED values, retrain the
value head, and test whether it finally learns joker value.

The value head was flat on joker-removal because single-trajectory labels are too
noisy (Part 1 verdict). Here we relabel each state with the MEAN of M bounded
rollouts (low variance) and retrain the ante head (MSE) on those labels. Then we
check, vs the old single-traj net, whether:
  - joker-removal Δ flips from ~0 to clearly POSITIVE (it values jokers now), and
  - val corr to held-out averaged labels beats the old ~0.47 ceiling.

    PYTHONPATH=src python scripts/phase8_value_relabel_retrain.py \
        --states 400 --rollouts 6 --max-antes 3 --epochs 40 --jobs 8 \
        --ckpt-out .data/phase8_value_relabel_v0.pt --metrics .data/phase8_relabel.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_NORM = 9.0  # label normalizer (ante reached, ~[1,9]) -> [0,1] for the sigmoid head


def _frac(state) -> float:
    if state.required_score and state.required_score > 0:
        return min(1.0, max(0.0, state.current_score / state.required_score))
    return 0.0


def _rollout_value(state, *, seed, rollout_bot, max_antes, max_steps=300):
    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import GamePhase
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    from dataclasses import replace as _replace

    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope

    sim = LocalBalatroSimulator(seed=seed, stake="white")
    sim.state = state
    bot = create_bot(rollout_bot, seed=seed)
    start_ante = state.ante
    # Disable the shop-decision audit in the rollout pilot — it ~doubles the late-shop
    # cost and is never read here (offline eval tools only).
    with bot_config_scope(_replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        for _ in range(max_steps):
            s = sim.state
            if s.won:
                return 9.0  # win dominates any non-win (max non-win ~8 + frac)
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


def _collect_states(seeds, cap, per_seed, min_ante=1, min_jokers=0):
    """Diverse states (the value head must evaluate any ante). Filters optional."""
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
        got = 0
        for _ in range(1500):
            st = sim.state
            if st.run_over or st.phase == GamePhase.RUN_OVER:
                break
            if (st.phase in (GamePhase.SELECTING_HAND, GamePhase.SHOP)
                    and st.ante >= min_ante and len(st.jokers) >= min_jokers):
                out.append(st)
                got += 1
                if len(out) >= cap:
                    return out
                if got >= per_seed:
                    break
            sim.step(drv.choose_action(st))
    return out


def _label_job(arg):
    """Average M bounded rollouts from a state -> low-variance value label."""
    st, si, m_rollouts, rollout_bot, max_antes = arg
    from balatro_ai.api.state import with_derived_legal_actions
    from balatro_ai.ml.encoding import encode_state

    st_d = with_derived_legal_actions(st)
    vals = [_rollout_value(st_d, seed=7919 * (m + 1) + 31 * si,
                           rollout_bot=rollout_bot, max_antes=max_antes)
            for m in range(m_rollouts)]
    return encode_state(st), statistics.mean(vals), st


def _net_joker_delta(model, states):
    import torch
    from balatro_ai.api.state import with_derived_legal_actions
    from balatro_ai.ml.encoding import encode_state
    from balatro_ai.ml.model import collate_states

    model.eval()
    deltas = []
    with torch.no_grad():
        for st in states:
            if len(st.jokers) < 2:
                continue
            base = float(model.ante_value(collate_states([encode_state(with_derived_legal_actions(st))]))[0])
            for i in range(len(st.jokers)):
                without = with_derived_legal_actions(dataclasses.replace(
                    st, jokers=tuple(j for k, j in enumerate(st.jokers) if k != i)))
                v = float(model.ante_value(collate_states([encode_state(without)]))[0])
                deltas.append(base - v)
    if not deltas:
        return 0.0, 0.0
    return statistics.mean(deltas), sum(1 for d in deltas if d > 0) / len(deltas)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=400)
    ap.add_argument("--rollouts", type=int, default=6)
    ap.add_argument("--max-antes", type=int, default=3)
    ap.add_argument("--per-seed", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--rollout-bot", default="basic_strategy_bot")
    ap.add_argument("--old-ckpt", default=".data/phase8_value_v0.pt")
    ap.add_argument("--ckpt-out", default=".data/phase8_value_relabel_v0.pt")
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    seeds = [f"{500000 + i:07d}" for i in range(1, 600)]
    states = _collect_states(seeds, cap=args.states, per_seed=args.per_seed)
    print(f"[relabel] collected {len(states)} states; labeling with mean of "
          f"{args.rollouts} (+{args.max_antes}-ante) rollouts...", flush=True)

    jobs = [(st, si, args.rollouts, args.rollout_bot, args.max_antes) for si, st in enumerate(states)]
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(_label_job, jobs))
    else:
        results = [_label_job(j) for j in jobs]
    encoded = [r[0] for r in results]
    labels = [r[1] for r in results]
    raw = [r[2] for r in results]
    print(f"[relabel] labeled {len(labels)}; label mean={statistics.mean(labels):.2f} "
          f"std={statistics.pstdev(labels):.2f}", flush=True)

    import torch
    from torch import nn

    from balatro_ai.ml.model import ValueNet, collate_states
    from balatro_ai.ml.train import _pearson, save_checkpoint

    torch.manual_seed(0)
    idx = list(range(len(encoded)))
    random.Random(0).shuffle(idx)
    n_val = max(1, len(idx) // 5)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    targets = [min(1.0, max(0.0, labels[i] / _NORM)) for i in range(len(labels))]

    model = ValueNet(dropout=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    mse = nn.MSELoss()
    bs = args.batch_size
    model.train()
    for _ in range(args.epochs):
        random.Random(_).shuffle(train_idx)
        for start in range(0, len(train_idx), bs):
            chunk = train_idx[start:start + bs]
            batch = collate_states([encoded[i] for i in chunk])
            tgt = torch.tensor([targets[i] for i in chunk], dtype=torch.float32)
            opt.zero_grad()
            loss = mse(model.ante_value(batch), tgt)
            loss.backward()
            opt.step()

    # eval: corr to held-out averaged labels
    model.eval()
    with torch.no_grad():
        val_pred = model.ante_value(collate_states([encoded[i] for i in val_idx])).tolist()
    val_lab = [labels[i] for i in val_idx]
    new_corr = _pearson(val_pred, val_lab)

    # joker-removal on a DEDICATED joker-rich eval set (ante>=4, jokers>=2) from
    # disjoint seeds — independent of the train/val split. new vs old net.
    from balatro_ai.ml.train import load_checkpoint
    eval_seeds = [f"{650000 + i:07d}" for i in range(1, 120)]
    eval_states = _collect_states(eval_seeds, cap=40, per_seed=2, min_ante=4, min_jokers=2)
    new_delta, new_frac = _net_joker_delta(model, eval_states)
    try:
        old_delta, old_frac = _net_joker_delta(load_checkpoint(args.old_ckpt), eval_states)
        old_delta, old_frac = round(old_delta, 4), round(old_frac, 3)
    except Exception as e:
        # Pre-fix checkpoints (ENCODING_VERSION 1) have a different item-vocab size and
        # were trained on the buggy encoder anyway, so load_checkpoint rejects them and
        # the comparison is moot. Skip cleanly rather than abort the whole run.
        print(f"[relabel] old-net comparison skipped ({type(e).__name__}: {e})", flush=True)
        old_delta, old_frac = None, None
    save_checkpoint(model, args.ckpt_out)

    out = {
        "n_states": len(states), "rollouts_per_state": args.rollouts,
        "max_antes": args.max_antes, "n_train": len(train_idx), "n_val": len(val_idx),
        "val_corr_to_avg_label": round(new_corr, 4),
        "joker_eval_states": len(eval_states),
        "new_net_joker_delta": round(new_delta, 4), "new_net_joker_frac_pos": round(new_frac, 3),
        "old_net_joker_delta": old_delta, "old_net_joker_frac_pos": old_frac,
        "ckpt": args.ckpt_out,
    }
    print("[relabel] RESULT:", json.dumps(out, indent=2), flush=True)
    with open(args.metrics, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    open(f".data/_RELABEL_corr{out['val_corr_to_avg_label']:.2f}"
         f"_newdelta{out['new_net_joker_delta']:.3f}_newfrac{out['new_net_joker_frac_pos']:.2f}", "w").close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
