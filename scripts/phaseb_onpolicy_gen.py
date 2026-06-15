"""On-policy iteration-1 data generation (Fable-5 plan, the exploration step).

Generates a FRESH capture set from the current (iteration-0 BC) policy, with the
plan's targeted exploration — the piece a pure offline AWR retrain cuts:

  - temperature-sample among the top-k candidates ONLY on decisions where the
    policy's top-2 margin is small (low confidence) AND the decision is
    recoverable (NOT a do-or-die play). Naive uniform temperature lowers the
    data-gen winrate and shrinks the winning mass the AWR tilt rides on, so
    exploration is gated to where a wrong turn is cheap.
  - ~20% of episodes are recipe episodes (RouteRecipeBot / flush-commit),
    continuing the iteration-0 diversity so winner-upweighting always has
    alternative lines to compare against (never a filter, only a tilt).

Honest-mode throughout: every observation is `blind_known_deck`-ed before the
policy sees it (identity no-op when BALATRO_NO_FORESIGHT is unset). Writes a
mixture-format JSONL (the same RunCapture.to_json_dict the iteration-0 mixture
used) so it concatenates directly with the iteration-0 data + fork-audit labels.

    PYTHONPATH=src BALATRO_NO_FORESIGHT=shuffle python scripts/phaseb_onpolicy_gen.py \
        --ckpt .data/phaseb_policy_b0v2.pt --n 2000 --seed-offset 5200000 \
        --recipe-frac 0.2 --margin 0.2 --temp 1.0 --topk 4 --jobs 12 \
        --out .data/phaseb_onpolicy_2000.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_RECIPES = (
    "recipe_flush_commit_bot",
    "recipe_buy_best_bot",
    "recipe_hoard2_spend_bot",
    "recipe_level_focus_bot",
    "recipe_reroll_hunt_bot",
    "recipe_skip_smalls_bot",
    "recipe_churn_weakest_bot",
    "recipe_buy_best_skip_bot",
)


class _ExplorationPolicy:
    """Neural policy with margin/recoverability-gated temperature exploration.

    Greedy (argmax) by default; on a low-confidence AND recoverable decision it
    temperature-samples among the top-k candidates. Any inference failure falls
    back to the heuristic — exploration can degrade a run, never crash it."""

    def __init__(self, ckpt: str, margin: float, temp: float, topk: int, rng_seed: int) -> None:
        from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
        from balatro_ai.ml.policy_net import load_policy

        self._net = load_policy(ckpt)
        self._fallback = BasicStrategyBot(seed=0)
        self._margin = margin
        self._temp = max(1e-3, temp)
        self._topk = max(1, topk)
        import random as _random

        self._rng = _random.Random(rng_seed)
        self.explore_decisions = 0
        self.total_decisions = 0

    def __call__(self, state):
        import torch

        from balatro_ai.api.actions import ActionType
        from balatro_ai.api.state import with_derived_legal_actions
        from balatro_ai.bots.no_foresight import blind_known_deck
        from balatro_ai.ml.dataset import candidate_tokens_for_state
        from balatro_ai.ml.encoding import encode_state
        from balatro_ai.ml.policy_net import candidate_logit_vector

        state = with_derived_legal_actions(blind_known_deck(state))
        self.total_decisions += 1
        try:
            legals = state.legal_actions
            cands = candidate_tokens_for_state(state)
            if not cands or len(cands) != len(legals):
                return self._fallback.choose_action(state)
            logits = candidate_logit_vector(self._net, encode_state(state), cands)
            probs = torch.softmax(logits, dim=0)
            order = torch.argsort(probs, descending=True)
            argmax_i = int(order[0])
            margin = float(probs[order[0]] - probs[order[1]]) if probs.numel() > 1 else 1.0
            if not self._should_explore(state, margin):
                sel = argmax_i
            else:
                self.explore_decisions += 1
                k = min(self._topk, int(order.numel()))
                topk_idx = order[:k]
                tp = torch.softmax(logits[topk_idx] / self._temp, dim=0)
                choice = self._rng.choices(range(k), weights=tp.tolist(), k=1)[0]
                sel = int(topk_idx[choice])
            action = legals[sel]
            if action is None or action.action_type == ActionType.NO_OP:
                return self._fallback.choose_action(state)
            return action
        except Exception:  # noqa: BLE001 — exploration must never crash a run
            return self._fallback.choose_action(state)

    def _should_explore(self, state, margin: float) -> bool:
        from balatro_ai.api.state import GamePhase

        if margin >= self._margin:  # confident -> stay greedy
            return False
        ph = state.phase
        # Recoverable surfaces: economy/route decisions can be undone over a run.
        if ph in (GamePhase.SHOP, GamePhase.BLIND_SELECT, GamePhase.BOOSTER_OPENED):
            return True
        # Play decisions only with a safety net (>=2 hands left), so a probe
        # play that whiffs never directly loses the blind (the must-clear gate).
        if ph == GamePhase.SELECTING_HAND:
            return int(getattr(state, "hands_remaining", 0)) >= 2
        return False


def _gen_one(task) -> dict | None:
    seed, stake, ckpt, recipe_name, margin, temp, topk, rng_seed = task
    from dataclasses import replace

    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.ml.dataset import capture_run

    if recipe_name is not None:
        from balatro_ai.bots.registry import create_bot

        # capture_run wants a callable; registry bots expose .choose_action
        # (and blind internally for honest mode — SolverShopBasicPlayBot /
        # _BlindedPolicyBot both call blind_known_deck), so the bound method is
        # the honest-consistent callable.
        policy = create_bot(recipe_name, seed=rng_seed).choose_action
        explore_frac = None
    else:
        policy = _ExplorationPolicy(ckpt, margin, temp, topk, rng_seed)
        explore_frac = policy  # keep ref for stats

    try:
        with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
            cap = capture_run(seed, policy, stake=stake)
    except Exception as exc:  # noqa: BLE001
        return {"seed": seed, "error": f"{type(exc).__name__}: {exc}"}
    d = cap.to_json_dict()
    d["bot_name"] = recipe_name or "neural_explore"
    if explore_frac is not None and explore_frac.total_decisions:
        d["_explore_rate"] = round(explore_frac.explore_decisions / explore_frac.total_decisions, 3)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="iteration-0 BC policy to generate from")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed-offset", type=int, default=5200000,
                    help="fresh seeds (disjoint from the 5.1M training/holdout ranges)")
    ap.add_argument("--stake", default="white")
    ap.add_argument("--recipe-frac", type=float, default=0.2)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import random

    rng = random.Random(args.seed_offset)
    tasks = []
    for i in range(1, args.n + 1):
        seed = f"{args.seed_offset + i:07d}"
        is_recipe = rng.random() < args.recipe_frac
        recipe = rng.choice(_RECIPES) if is_recipe else None
        tasks.append((seed, args.stake, args.ckpt, recipe, args.margin, args.temp, args.topk,
                      args.seed_offset + i))
    n_recipe = sum(1 for t in tasks if t[3] is not None)
    print(f"[onpolicy] {args.n} episodes ({args.n - n_recipe} neural-explore + {n_recipe} recipe); "
          f"margin<{args.margin}, temp={args.temp}, top{args.topk}", flush=True)

    rows: list[dict] = []
    n_won = 0
    n_err = 0
    explore_rates: list[float] = []
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=args.jobs) as ex, open(args.out, "w", encoding="utf-8") as fh:
        for i, d in enumerate(ex.map(_gen_one, tasks), start=1):
            if d is None:
                continue
            if d.get("error"):
                n_err += 1
            else:
                fh.write(json.dumps(d) + "\n")
                rows.append(d)
                if d.get("won"):
                    n_won += 1
                if "_explore_rate" in d:
                    explore_rates.append(d["_explore_rate"])
            if i % 200 == 0:
                wr = n_won / max(1, len(rows))
                print(f"[onpolicy] {i}/{args.n}: {len(rows)} captured, winrate {wr:.1%}, {n_err} errors",
                      flush=True)
    wr = n_won / max(1, len(rows))
    import statistics

    mean_expl = statistics.mean(explore_rates) if explore_rates else 0.0
    print(f"[onpolicy] DONE: {len(rows)} captured ({n_won} won, {wr:.1%}), {n_err} errors; "
          f"mean explore-rate on neural episodes {mean_expl:.1%} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
