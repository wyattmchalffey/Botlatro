"""Full-fidelity Phase-B iteration 1 (Fable-5 plan, no corners cut).

This is the iteration-1 test the plan actually specifies — the minimal offline
AWR run (phaseb_iter1.py) was a mechanism check that cut four corners; this
restores all of them:

  1. RICH per-candidate fusion features (hand_strength / is_discard / draw_odds,
     schema v2) — already in the re-expanded cache this reads.
  2. DENSE fork-audit play labels: the d6w6 beam's clearing lines at death
     blinds (search-improved targets) merged in at an ELEVATED flat weight.
  3. ON-POLICY data: a fresh capture set generated from the iter-0 policy with
     margin/recoverability-gated exploration (+ 20% recipe episodes), merged
     with the iteration-0 mixture.
  4. PER-POLICY-baseline AWR on the behavior data (preserves recipe diversity)
     + a POWERED 1024-2048 paired gate reporting d_winrate CI, McNemar p, AND
     the mean-ante surrogate.

Behavior examples (mixture + on-policy) are advantage-weighted; fork-audit
labels are search-verified wins, not behavior, so they get a flat elevated
weight (AWR's R-V baseline doesn't apply to them).

    PYTHONPATH=src BALATRO_NO_FORESIGHT=shuffle python scripts/phaseb_iter1_full.py \
        --mixture .data/phaseb_mix_1000.jsonl --onpolicy .data/phaseb_onpolicy_2000.jsonl \
        --forkaudit .data/phaseb_forkaudit_labels.pkl --fork-weight 5.0 \
        --heldout .data/phaseb_heldout_200.jsonl --baseline-value .data/phaseb_policy_v0salvage.pt \
        --b0 .data/phaseb_policy_b0v2.pt --beta 2.0 --per-policy-blend 0.5 \
        --epochs 15 --eval-seeds 1024 --eval-offset 5300000 --jobs 12 \
        --out .data/phaseb_policy_iter1full.pt --result .data/phaseb_iter1full_result.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _eval_seed(task) -> tuple[bool, int]:
    """Drive `neural_policy_bot` (ckpt via env) on one seed; return (won, ante)."""
    seed, ckpt = task
    # CPU eval: forked pool worker + the parent's post-training CUDA context = a
    # deadlock if the worker touches CUDA. Single-game inference needs no GPU.
    os.environ["BALATRO_DEVICE"] = "cpu"
    from dataclasses import replace

    # Deterministic eval: single-threaded torch fixes the FP reduction order, so a
    # fixed policy on a fixed seed is bit-reproducible. Without this, multi-threaded
    # reductions flip ~1-2% of argmax-boundary decisions run-to-run, and over a
    # 4000-step game that compounds into a different win/loss outcome — so the SAME
    # policy benched twice gave 38 vs 42 wins / 1024, an irreproducible point
    # estimate. (Single-process back-to-back is already 0/24; the sim is
    # deterministic per seed — this is purely the parallel-eval threading layer.)
    # The pool parallelizes across PROCESSES, so one thread per worker keeps all
    # cores busy at zero throughput cost.
    import random

    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    from balatro_ai.api.state import with_derived_legal_actions
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    os.environ["BALATRO_POLICY_CKPT"] = ckpt
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
    bot = create_bot("neural_policy_bot", seed=0)
    with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        for _ in range(4000):
            st = sim.state
            if st.run_over:
                break
            a = bot.choose_action(st)
            if a.action_type.value == "no_op":
                break
            try:
                sim.step(a)
            except Exception:  # noqa: BLE001
                break
    return bool(sim.state.won), int(sim.state.ante)


def _bench(ckpt: str, seeds: list[str], workers: int) -> list[tuple[bool, int]]:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_eval_seed, [(s, ckpt) for s in seeds]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixture", required=True)
    ap.add_argument("--onpolicy", default=None, help="on-policy capture JSONL (optional but planned)")
    ap.add_argument("--forkaudit", default=None, help="pickle of fork-audit TrainingExamples")
    ap.add_argument("--fork-weight", type=float, default=5.0)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--baseline-value", required=True, help="frozen V0 value-head ckpt (AWR baseline)")
    ap.add_argument("--b0", required=True, help="iteration-0 policy to gate against")
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--w-max", type=float, default=5.0)
    ap.add_argument("--per-policy-blend", type=float, default=0.5)
    ap.add_argument("--progress-lambda", type=float, default=0.0,
                    help="AWR reward shaping: R=(1-L)*won+L*ante_norm (0=binary win/loss, "
                         "L>0 credits far-but-losing runs by how far they reached)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--eval-seeds", type=int, default=1024)
    ap.add_argument("--eval-offset", type=int, default=5300000)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", default=".data/phaseb_policy_iter1full.pt")
    ap.add_argument("--result", default=".data/phaseb_iter1full_result.json")
    ap.add_argument("--shard-dir", default=None,
                    help="out-of-core base dir: streams mixture+on-policy from disk shards (50k+ runs)")
    ap.add_argument("--runs-per-shard", type=int, default=500)
    args = ap.parse_args()

    from balatro_ai.bench_stats import mcnemar_exact_p, paired_delta_ci, paired_mean_diff_ci
    from balatro_ai.ml.dataset import load_or_expand_examples
    from balatro_ai.ml.policy_net import (
        PolicyConfig,
        compute_advantage_weights,
        load_policy,
        save_policy,
        train_decision_policy,
    )

    # ---- common: fork labels (small, in RAM), held-out, AWR baseline ------ #
    from balatro_ai.ml.policy_net import _labelled

    fork: list = []
    if args.forkaudit and os.path.exists(args.forkaudit):
        with open(args.forkaudit, "rb") as fh:
            fork = pickle.load(fh)
        print(f"[iter1full] + {len(fork)} fork-audit search-improved play labels "
              f"(flat weight {args.fork_weight})", flush=True)
    # held-out `val` is loaded LATER, right before training — at 50k scale it is
    # ~40GB in RAM, and keeping it out during the parallel AWR pass leaves headroom
    # for the AWR worker pool.
    baseline = load_policy(args.baseline_value)
    cfg = PolicyConfig(epochs=args.epochs, weight_decay=1e-3, dropout=0.2, seed=0)

    if args.shard_dir:
        # ---- out-of-core: stream mixture+on-policy from disk shards -------- #
        from balatro_ai.ml.dataset import ShardedExampleStore, expand_to_shards
        from balatro_ai.ml.policy_net import (
            compute_advantage_weights_sharded,
            train_decision_policy_sharded,
        )

        mix_dir = os.path.join(args.shard_dir, "mix")
        expand_to_shards(args.mixture, mix_dir, runs_per_shard=args.runs_per_shard)
        dirs = [mix_dir]
        if args.onpolicy and os.path.exists(args.onpolicy):
            onp_dir = os.path.join(args.shard_dir, "onp")
            expand_to_shards(args.onpolicy, onp_dir, runs_per_shard=args.runs_per_shard)
            dirs.append(onp_dir)
        store = ShardedExampleStore(dirs)
        weight_dir = os.path.join(args.shard_dir, "awr_w")
        print(f"[iter1full] out-of-core: {len(store)} behavior examples in "
              f"{len(store.shard_files)} shards + {len(fork)} fork; AWR weights...", flush=True)
        compute_advantage_weights_sharded(
            store, baseline, weight_dir, beta=args.beta, w_max=args.w_max,
            per_policy_blend=args.per_policy_blend, progress_lambda=args.progress_lambda,
            baseline_ckpt=args.baseline_value,
        )
        val = load_or_expand_examples(args.heldout)  # deferred until after the AWR pass
        net, m = train_decision_policy_sharded(
            store, cfg, val_examples=val, weight_dir=weight_dir,
            extra_examples=fork, extra_weight=args.fork_weight,
        )
    else:
        # ---- in-RAM path (local / small datasets) ------------------------- #
        behavior = load_or_expand_examples(args.mixture)
        n_mix = len(behavior)
        if args.onpolicy and os.path.exists(args.onpolicy):
            onp = load_or_expand_examples(args.onpolicy)
            behavior = behavior + onp
            print(f"[iter1full] mixture {n_mix} + on-policy {len(onp)} = {len(behavior)} behavior",
                  flush=True)
        else:
            print(f"[iter1full] mixture {n_mix} behavior examples (no on-policy set)", flush=True)
        w_behavior = compute_advantage_weights(
            behavior, baseline, beta=args.beta, w_max=args.w_max,
            per_policy_blend=args.per_policy_blend, progress_lambda=args.progress_lambda,
        )
        lab_fork = _labelled(fork)
        all_examples = behavior + fork
        all_weights = w_behavior + [args.fork_weight] * len(lab_fork)
        import statistics
        print(f"[iter1full] labelled: {len(_labelled(behavior))} behavior + {len(lab_fork)} fork; "
              f"AWR weight mean={statistics.mean(w_behavior):.3f} max={max(w_behavior):.2f}", flush=True)
        val = load_or_expand_examples(args.heldout)
        net, m = train_decision_policy(all_examples, cfg, val_examples=val, example_weights=all_weights)

    print(f"[iter1full] policy: top1={m['top1']} top1_no_heuristic={m['top1_no_heuristic']} "
          f"value_auc={m.get('best_val_value_auc')}", flush=True)
    save_policy(net, args.out, config=cfg)

    # ---- powered paired gate vs B0 --------------------------------------- #
    seeds = [f"{args.eval_offset + i:07d}" for i in range(1, args.eval_seeds + 1)]
    print(f"[iter1full] POWERED GATE: {len(seeds)} paired seeds, iter1 vs B0 ...", flush=True)
    iter_res = _bench(args.out, seeds, args.jobs)
    b0_res = _bench(args.b0, seeds, args.jobs)
    n = len(seeds)
    iter_w = sum(1 for w, _ in iter_res if w)
    b0_w = sum(1 for w, _ in b0_res if w)
    gained = sum(1 for (a, _), (b, _) in zip(iter_res, b0_res) if a and not b)
    lost = sum(1 for (a, _), (b, _) in zip(iter_res, b0_res) if b and not a)
    p = mcnemar_exact_p(gained, lost)
    lo, hi = paired_delta_ci(gained, lost, n)
    ante_diffs = [ia - ba for (_, ia), (_, ba) in zip(iter_res, b0_res)]
    amean, alo, ahi = paired_mean_diff_ci(ante_diffs)

    result = {
        "n_seeds": n, "eval_offset": args.eval_offset,
        "iter1_wins": iter_w, "iter1_winrate": round(iter_w / n, 4),
        "b0_wins": b0_w, "b0_winrate": round(b0_w / n, 4),
        "gained": gained, "lost": lost,
        "d_winrate": round((gained - lost) / n, 4),
        "d_winrate_ci": [round(lo, 4), round(hi, 4)],
        "mcnemar_p": round(p, 4),
        "mean_ante_delta": round(amean, 4),
        "mean_ante_delta_ci": [round(alo, 4), round(ahi, 4)],
        "top1": m["top1"], "top1_no_heuristic": m["top1_no_heuristic"],
        "value_auc": m.get("best_val_value_auc"),
        "config": {"beta": args.beta, "w_max": args.w_max, "per_policy_blend": args.per_policy_blend,
                   "progress_lambda": args.progress_lambda,
                   "fork_weight": args.fork_weight, "epochs": args.epochs,
                   "n_behavior": m.get("n_examples", 0), "n_fork": len(fork),
                   "out_of_core": bool(args.shard_dir)},
    }
    print(f"[iter1full] iter1 {iter_w}/{n} ({iter_w/n:.1%}) vs B0 {b0_w}/{n} ({b0_w/n:.1%})", flush=True)
    print(f"[iter1full] d_winrate {(gained-lost)/n:+.1%} (95% CI {lo:+.1%}..{hi:+.1%}), "
          f"McNemar p={p:.3f}; gained {gained} / lost {lost}", flush=True)
    print(f"[iter1full] mean-ante surrogate {amean:+.3f} (95% CI {alo:+.3f}..{ahi:+.3f})", flush=True)
    # Single-iteration read (the cumulative 2-iter kill clause sums these JSONs).
    sig = lo > 0.0 or alo > 0.0
    verdict = "IMPROVES" if (gained > lost and sig) else (
        "NEUTRAL" if -0.02 <= (gained - lost) / n <= 0.02 else "REGRESSES")
    result["verdict"] = verdict
    print(f"[iter1full] VERDICT: {verdict} (winrate-CI-excludes-0={lo>0.0}, ante-CI-excludes-0={alo>0.0})",
          flush=True)
    print("[iter1full] NOTE: the plan's kill clause is CUMULATIVE over 2 iterations — "
          "this single read is one input, not the kill decision.", flush=True)
    with open(args.result, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"[iter1full] result -> {args.result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
