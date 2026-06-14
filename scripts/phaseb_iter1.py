"""Iteration 1 mechanism-check: does advantage-weighting beat plain BC?

The first outcome-tilted training, done OFFLINE on the cached mixture data as
the cheapest possible test of the improvement operator before paying for
on-policy generation. AWR weights (compute_advantage_weights, frozen salvaged
value head as baseline) upweight winning-trajectory decisions; the retrained
policy is benched PAIRED against the B0 (plain-BC) policy on held-out seeds.

Directional signal (AWR > B0) = the operator moves winrate -> justify scaled
on-policy generation (paid). Flat/negative = diagnose before spending.

    PYTHONPATH=src BALATRO_NO_FORESIGHT=shuffle python scripts/phaseb_iter1.py \
        --dataset .data/phaseb_mix_1000.jsonl --heldout .data/phaseb_heldout_200.jsonl \
        --baseline .data/phaseb_policy_v0salvage.pt --b0 .data/phaseb_policy_b0v2.pt \
        --beta 2.0 --epochs 15 --eval-seeds 256 --eval-offset 5101000
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _eval_seed(args) -> bool:
    seed, ckpt = args
    from dataclasses import replace

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
    return bool(sim.state.won)


def _bench(ckpt: str, seeds: list[str], workers: int) -> list[bool]:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_eval_seed, [(s, ckpt) for s in seeds]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--baseline", required=True, help="frozen value-head ckpt (salvaged V0)")
    ap.add_argument("--b0", required=True, help="plain-BC policy ckpt to beat")
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--w-max", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--eval-seeds", type=int, default=256)
    ap.add_argument("--eval-offset", type=int, default=5101000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--ckpt", default=".data/phaseb_policy_iter1.pt")
    args = ap.parse_args()

    from balatro_ai.bench_stats import mcnemar_exact_p, paired_delta_ci
    from balatro_ai.ml.dataset import load_or_expand_examples
    from balatro_ai.ml.policy_net import (
        PolicyConfig,
        compute_advantage_weights,
        load_policy,
        save_policy,
        train_decision_policy,
    )

    examples = load_or_expand_examples(args.dataset)
    val = load_or_expand_examples(args.heldout)
    baseline = load_policy(args.baseline)
    print(f"[iter1] {len(examples)} examples; computing AWR weights (beta={args.beta})...", flush=True)
    weights = compute_advantage_weights(examples, baseline, beta=args.beta, w_max=args.w_max)
    import statistics
    print(f"[iter1] weight mean={statistics.mean(weights):.3f} "
          f"max={max(weights):.2f} frac>1={sum(1 for w in weights if w > 1)/len(weights):.2f}",
          flush=True)

    cfg = PolicyConfig(epochs=args.epochs, weight_decay=1e-3, dropout=0.2, seed=0)
    net, m = train_decision_policy(examples, cfg, val_examples=val, example_weights=weights)
    print(f"[iter1] AWR policy: top1={m['top1']} value_auc={m.get('best_val_value_auc')}", flush=True)
    save_policy(net, args.ckpt, config=cfg)

    seeds = [f"{args.eval_offset + i:07d}" for i in range(1, args.eval_seeds + 1)]
    print(f"[iter1] benching AWR vs B0 on {len(seeds)} paired seeds...", flush=True)
    awr = _bench(args.ckpt, seeds, args.workers)
    b0 = _bench(args.b0, seeds, args.workers)
    n = len(seeds)
    awr_w, b0_w = sum(awr), sum(b0)
    gained = sum(1 for a, b in zip(awr, b0) if a and not b)
    lost = sum(1 for a, b in zip(awr, b0) if b and not a)
    p = mcnemar_exact_p(gained, lost)
    lo, hi = paired_delta_ci(gained, lost, n)
    print(f"[iter1] AWR {awr_w}/{n} ({awr_w/n:.1%}) vs B0 {b0_w}/{n} ({b0_w/n:.1%})", flush=True)
    print(f"[iter1] paired: gained {gained} / lost {lost}, d_winrate {(gained-lost)/n:+.1%} "
          f"(95% CI {lo:+.1%}..{hi:+.1%}), McNemar p={p:.3f}", flush=True)
    direction = "AWR HIGHER" if awr_w > b0_w else ("TIED" if awr_w == b0_w else "AWR LOWER")
    print(f"[iter1] MECHANISM READ: {direction} — "
          f"{'operator moves winrate, justify scaled on-policy generation' if awr_w > b0_w else 'no offline gain; diagnose before paid scale'}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
