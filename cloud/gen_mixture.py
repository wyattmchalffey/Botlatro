"""Sharded, resumable iteration-0 mixture generation for cloud Phase B.

Composition per PHASE_B_ARCHITECTURE.md (the one my local proxy under-weighted —
it was recipe-heavy at 7.6%; this restores the planned ~12% base):
  ~55% deployed solver  (solver_shop_basic_play_bot, the ~12-14% baseline)
  ~35% recipe episodes  (the 8 route recipes, for shop/route diversity)
  ~10% play-perturbation (deployed bot, but SELECTING_HAND samples among the
       top-k plays instead of argmax — the play-label diversity the plan needs,
       gated to >=2 hands remaining so a probe never throws a must-clear blind)

CLOUD-CRITICAL properties:
  - SHARDABLE: --shards N --shard-id i; shard i owns seeds with idx%N==i, so K
    nodes cover one seed range with zero overlap and near-linear speedup.
  - RESUMABLE: appends to out-dir/shard{id}.jsonl and skips seeds already there,
    so a spot-preemption resumes instead of restarting (each shard is its own
    durable checkpoint).
  - HONEST: BALATRO_NO_FORESIGHT=shuffle; every bot blinds the observation
    internally (solver/recipe both call blind_known_deck).

    PYTHONPATH=src BALATRO_NO_FORESIGHT=shuffle python cloud/gen_mixture.py \
        --n-runs 50000 --seed-offset 5100000 --shards 1 --shard-id 0 \
        --jobs 0 --out-dir .data/cloud_mix
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_RECIPES = (
    "recipe_flush_commit_bot", "recipe_buy_best_bot", "recipe_hoard2_spend_bot",
    "recipe_level_focus_bot", "recipe_reroll_hunt_bot", "recipe_skip_smalls_bot",
    "recipe_churn_weakest_bot", "recipe_buy_best_skip_bot",
)
_DEPLOYED = "solver_shop_basic_play_bot"


def _kind_for_seed(seed: str, deployed_frac: float, recipe_frac: float) -> str:
    """Deterministic generator assignment by hashing the seed (reproducible +
    balanced across any shard split)."""
    h = int(hashlib.blake2b(seed.encode(), digest_size=8).hexdigest(), 16) / 2**64
    if h < deployed_frac:
        return "deployed"
    if h < deployed_frac + recipe_frac:
        return "recipe"
    return "perturb"


class _PerturbPlayBot:
    """Deployed solver, but SELECTING_HAND plays are sampled from the top-k by
    immediate Rust score (the plan's play-diversity source). Everything else
    (shop, blind-select) delegates to the deployed bot at full quality."""

    def __init__(self, seed: int, topk: int, rng_seed: int) -> None:
        from balatro_ai.bots.registry import create_bot

        self._bot = create_bot(_DEPLOYED, seed=seed)
        self._topk = max(2, topk)
        import random as _random

        self._rng = _random.Random(rng_seed)

    def choose_action(self, state):
        from balatro_ai.api.actions import ActionType
        from balatro_ai.api.state import GamePhase, with_derived_legal_actions
        from balatro_ai.bots.no_foresight import blind_known_deck

        if state.phase != GamePhase.SELECTING_HAND:
            return self._bot.choose_action(state)
        # Only perturb when there's a safety net (>=2 hands): a probe play that
        # whiffs never directly loses the blind.
        if int(getattr(state, "hands_remaining", 0)) < 2:
            return self._bot.choose_action(state)
        try:
            obs = with_derived_legal_actions(blind_known_deck(state))
            legals = obs.legal_actions
            plays = [(i, a) for i, a in enumerate(legals)
                     if a.action_type == ActionType.PLAY_HAND and a.card_indices]
            if len(plays) < 2:
                return self._bot.choose_action(state)
            from balatro_ai.search.rust_bridge import rust_play_action_evals

            evals = rust_play_action_evals(obs, [a for _, a in plays])
            if not evals:
                return self._bot.choose_action(state)
            scored = [(plays[j][1], (evals[j][0] if evals[j] else 0))
                      for j in range(len(plays))]
            scored.sort(key=lambda t: t[1], reverse=True)
            k = min(self._topk, len(scored))
            return self._rng.choice(scored[:k])[0]
        except Exception:  # noqa: BLE001 — perturbation must never crash a run
            return self._bot.choose_action(state)


def _gen_one(task) -> dict | None:
    seed, stake, kind, topk, rng_seed = task
    from dataclasses import replace

    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.ml.dataset import capture_run

    if kind == "deployed":
        from balatro_ai.bots.registry import create_bot
        policy = create_bot(_DEPLOYED, seed=rng_seed).choose_action
        name = _DEPLOYED
    elif kind == "recipe":
        from balatro_ai.bots.registry import create_bot
        # pick a recipe deterministically from the seed
        ri = int(hashlib.blake2b((seed + "r").encode(), digest_size=4).hexdigest(), 16) % len(_RECIPES)
        name = _RECIPES[ri]
        policy = create_bot(name, seed=rng_seed).choose_action
    else:  # perturb
        policy = _PerturbPlayBot(rng_seed, topk, rng_seed).choose_action
        name = "deployed_perturb"

    try:
        with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
            cap = capture_run(seed, policy, stake=stake)
    except Exception as exc:  # noqa: BLE001
        return {"seed": seed, "error": f"{type(exc).__name__}: {exc}"}
    d = cap.to_json_dict()
    d["bot_name"] = name
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, required=True, help="total runs across ALL shards")
    ap.add_argument("--seed-offset", type=int, default=5100000)
    ap.add_argument("--stake", default="white")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--deployed-frac", type=float, default=0.55)
    ap.add_argument("--recipe-frac", type=float, default=0.35)
    ap.add_argument("--perturb-topk", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=0, help="0 = os.cpu_count()")
    ap.add_argument("--out-dir", default=".data/cloud_mix")
    ap.add_argument("--flush-every", type=int, default=50)
    args = ap.parse_args()

    jobs = args.jobs or (os.cpu_count() or 4)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"shard{args.shard_id}.jsonl")

    # This shard owns the seeds whose 0-based index % shards == shard_id.
    all_idx = range(1, args.n_runs + 1)
    my_seeds = [f"{args.seed_offset + i:07d}" for i in all_idx if (i - 1) % args.shards == args.shard_id]

    # Resume: skip seeds already captured in this shard's file.
    done: set[str] = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        done.add(str(json.loads(line)["seed"]))
                    except Exception:  # noqa: BLE001
                        pass
    todo = [s for s in my_seeds if s not in done]
    print(f"[gen] shard {args.shard_id}/{args.shards}: {len(my_seeds)} seeds, "
          f"{len(done)} already done, {len(todo)} to generate; jobs={jobs}; "
          f"mix deployed={args.deployed_frac} recipe={args.recipe_frac} "
          f"perturb={1-args.deployed_frac-args.recipe_frac:.2f}", flush=True)

    tasks = [(s, args.stake, _kind_for_seed(s, args.deployed_frac, args.recipe_frac),
              args.perturb_topk, args.seed_offset + int(s)) for s in todo]
    n_done = len(done)
    n_won = 0
    n_err = 0
    from collections import Counter
    kinds = Counter(t[2] for t in tasks)
    print(f"[gen] this-shard kinds: {dict(kinds)}", flush=True)

    with ProcessPoolExecutor(max_workers=jobs) as ex, open(out_path, "a", encoding="utf-8") as fh:
        for i, d in enumerate(ex.map(_gen_one, tasks), start=1):
            if d is None:
                continue
            if d.get("error"):
                n_err += 1
            else:
                fh.write(json.dumps(d) + "\n")
                if d.get("won"):
                    n_won += 1
            if i % args.flush_every == 0:
                fh.flush()
                os.fsync(fh.fileno())  # durable checkpoint for spot-resume
                tot = n_done + i
                print(f"[gen] shard {args.shard_id}: {i}/{len(tasks)} new "
                      f"(winrate {n_won}/{max(1,i-n_err):.0f} ~ {n_won/max(1,i-n_err):.1%}), "
                      f"{n_err} errors, {tot} total in shard", flush=True)
    print(f"[gen] shard {args.shard_id} DONE: +{len(tasks)-n_err} captured "
          f"({n_won} won, {n_won/max(1,len(tasks)-n_err):.1%}), {n_err} errors -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
