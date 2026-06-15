"""Extend the iteration-1 paired gate to 2048 seeds (the plan's powered upper
bound) by benching a FRESH disjoint seed block and COMBINING discordant counts.

Paired McNemar counts are additive across disjoint seed sets, so a fresh 1024
block adds to the first gate's (gained=34, lost=22) for a 2048-seed read — the
decisive test of whether iteration-1's +1.2pp lean is real or noise, for ~the
cost of one extra bench (no retraining).

    PYTHONPATH=src BALATRO_NO_FORESIGHT=shuffle python scripts/phaseb_gate_extend.py \
        --iter .data/phaseb_policy_iter1full.pt --b0 .data/phaseb_policy_b0rich.pt \
        --eval-seeds 1024 --eval-offset 5400000 \
        --prior-gained 34 --prior-lost 22 --prior-n 1024 --jobs 12
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _eval_seed(task) -> tuple[bool, int]:
    seed, ckpt = task
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
    return bool(sim.state.won), int(sim.state.ante)


def _bench(ckpt: str, seeds: list[str], workers: int) -> list[tuple[bool, int]]:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_eval_seed, [(s, ckpt) for s in seeds]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", required=True)
    ap.add_argument("--b0", required=True)
    ap.add_argument("--eval-seeds", type=int, default=1024)
    ap.add_argument("--eval-offset", type=int, default=5400000)
    ap.add_argument("--prior-gained", type=int, default=0)
    ap.add_argument("--prior-lost", type=int, default=0)
    ap.add_argument("--prior-n", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()

    from balatro_ai.bench_stats import mcnemar_exact_p, paired_delta_ci, paired_mean_diff_ci

    seeds = [f"{args.eval_offset + i:07d}" for i in range(1, args.eval_seeds + 1)]
    print(f"[extend] benching {len(seeds)} FRESH paired seeds (offset {args.eval_offset})...", flush=True)
    it = _bench(args.iter, seeds, args.jobs)
    b0 = _bench(args.b0, seeds, args.jobs)
    g = sum(1 for (a, _), (b, _) in zip(it, b0) if a and not b)
    l = sum(1 for (a, _), (b, _) in zip(it, b0) if b and not a)
    it_w, b0_w = sum(1 for w, _ in it if w), sum(1 for w, _ in b0 if w)
    ante_diffs = [ia - ba for (_, ia), (_, ba) in zip(it, b0)]
    print(f"[extend] NEW block: iter {it_w}/{len(seeds)} vs B0 {b0_w}/{len(seeds)}; gained {g} / lost {l}",
          flush=True)

    # Combine with the prior gate's discordant counts for the 2048-seed read.
    G, L, N = g + args.prior_gained, l + args.prior_lost, len(seeds) + args.prior_n
    p = mcnemar_exact_p(G, L)
    lo, hi = paired_delta_ci(G, L, N)
    amean, alo, ahi = paired_mean_diff_ci(ante_diffs)  # new block only (ante surrogate)
    print(f"[extend] ===== COMBINED {N}-seed paired gate =====", flush=True)
    print(f"[extend] gained {G} / lost {L}  ->  d_winrate {(G-L)/N:+.2%} "
          f"(95% CI {lo:+.2%}..{hi:+.2%}), McNemar p={p:.4f}", flush=True)
    sig = lo > 0.0
    print(f"[extend] VERDICT: {'SIGNIFICANT POSITIVE' if (G>L and sig) else 'still inconclusive (CI includes 0)'}",
          flush=True)
    print(f"[extend] (new-block mean-ante {amean:+.3f}, CI {alo:+.3f}..{ahi:+.3f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
