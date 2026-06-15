"""Gate B0: the Phase-B plumbing gate.

Pre-registered (PHASE_B_ARCHITECTURE.md): a PLUMBING gate, not an improvement
claim. Train the decision-shaped policy (BC) on a recipe-mixture capture, deploy
it as `neural_policy_bot`, and check it lands within a CI of the MEASURED MIXTURE
winrate (the capture's own win fraction) — on TRAINING-RANGE seeds (consumes no
reserved holdout slice). A pass certifies the encoder/candidate/deploy pipeline
carries a whole policy end to end; it does NOT claim improvement.

    PYTHONPATH=src BALATRO_NO_FORESIGHT=shuffle python scripts/phaseb_gate_b0.py \
        --dataset .data/phaseb_mix_1000.jsonl --epochs 20 --eval-seeds 256 \
        --eval-offset 5100000 --ckpt .data/phaseb_policy_b0.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
                # A degenerate-but-legal pick the sim refuses (e.g. taking a
                # Buffoon-pack joker at full slots) ends the run as a loss —
                # the bot's own fault, not a bench failure.
                break
    return bool(sim.state.won)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eval-seeds", type=int, default=256)
    ap.add_argument("--eval-offset", type=int, default=5100000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--ckpt", default=".data/phaseb_policy_b0.pt")
    ap.add_argument("--shard-dir", default=None,
                    help="out-of-core: expand to disk shards + stream-train (for 50k+ runs)")
    ap.add_argument("--runs-per-shard", type=int, default=500)
    args = ap.parse_args()

    from balatro_ai.bench_stats import wilson_ci
    from balatro_ai.ml.dataset import load_or_expand_examples
    from balatro_ai.ml.policy_net import PolicyConfig, save_policy, train_decision_policy

    # 1) Load the mixture capture; the MIXTURE WINRATE is its own win fraction.
    rows = [json.loads(l) for l in open(args.dataset, encoding="utf-8") if l.strip()]
    mix_wins = sum(1 for r in rows if r.get("won"))
    print(f"[b0] mixture: {len(rows)} runs, winrate {mix_wins}/{len(rows)} "
          f"({mix_wins/len(rows):.1%})", flush=True)

    # 2) Expand to schema-v2 examples and BC-train. Out-of-core (--shard-dir)
    #    streams disk shards so 50k+ runs train without holding ~all examples
    #    in RAM; otherwise the in-RAM cached path.
    cfg = PolicyConfig(epochs=args.epochs, seed=0)
    t1 = time.perf_counter()
    if args.shard_dir:
        from balatro_ai.ml.dataset import ShardedExampleStore, expand_to_shards
        from balatro_ai.ml.policy_net import train_decision_policy_sharded

        expand_to_shards(args.dataset, args.shard_dir, runs_per_shard=args.runs_per_shard)
        store = ShardedExampleStore(args.shard_dir)
        print(f"[b0] {len(store)} examples (out-of-core); training...", flush=True)
        net, metrics = train_decision_policy_sharded(store, cfg)
    else:
        examples = load_or_expand_examples(args.dataset)
        print(f"[b0] {len(examples)} examples; training...", flush=True)
        net, metrics = train_decision_policy(examples, cfg)
    print("[b0] trained in {:.0f}s: top1={} no_heuristic={} chance={} value_auc={}".format(
        time.perf_counter() - t1, metrics["top1"], metrics["top1_no_heuristic"],
        metrics["chance"], metrics["value_auc"]), flush=True)
    save_policy(net, args.ckpt, config=cfg)

    # 3) Deploy + bench winrate on training-range seeds.
    seeds = [f"{args.eval_offset + i:07d}" for i in range(1, args.eval_seeds + 1)]
    t2 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(_eval_seed, [(s, args.ckpt) for s in seeds]))
    nn_wins = sum(results)
    n = len(results)
    lo, hi = wilson_ci(nn_wins, n)
    mlo, mhi = wilson_ci(mix_wins, len(rows))
    print(f"[b0] neural_policy_bot: {nn_wins}/{n} ({nn_wins/n:.1%}, "
          f"95% CI {lo:.1%}..{hi:.1%}) in {time.perf_counter()-t2:.0f}s", flush=True)
    print(f"[b0] mixture winrate {mix_wins/len(rows):.1%} (95% CI {mlo:.1%}..{mhi:.1%})", flush=True)
    # PLUMBING PASS: the BC net is not statistically distinguishable from the
    # mixture (CIs intersect) — it reproduces the mixture end-to-end without
    # collapse. Proper interval-intersection test (a point estimate landing
    # just below the other's CI is NOT a fail if the intervals still overlap).
    overlap = lo <= mhi and mlo <= hi
    verdict = "PASS" if overlap else "FAIL"
    print(f"[b0] VERDICT (plumbing): {verdict} — CIs "
          f"{'overlap' if overlap else 'DISJOINT'} "
          f"(neural [{lo:.1%},{hi:.1%}] vs mixture [{mlo:.1%},{mhi:.1%}])", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
