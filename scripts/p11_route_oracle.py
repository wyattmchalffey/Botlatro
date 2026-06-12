"""P1.1 route-oracle: can ANY multi-shop route flip an honest loss?

The decisive go/no-go for the whole-run-planning thesis (SUPERHUMAN_ROADMAP
P1.1). For each honest-mode LOST seed: replay in faithful mode, fork at the
shops 1-2 antes before death, and roll each fork to terminal under a GRAMMAR
of route policies — deterministic multi-shop recipes (buy-best-every-shop,
economy hoard, leveling focus, reroll hunting, blind skipping, slot churn...).
A seed is ROUTABLE if any (fork, policy) rollout wins where the baseline lost.

This is hindsight-best-of-K (the S0 archetype-oracle pattern, extended from
4 archetypes to route recipes): it needs no future-reading planner and gives
a LOWER bound on the clairvoyant routing ceiling. Controls per the
kill-switch methodology: a NULL arm (fork + unmodified bot) establishes the
fork-noise floor; every rollout records `_rng_diverged` cleanliness; the
verdict is routable-above-null on the CLEAN subset.

PRE-REGISTERED GATE: if <15% of losses flip above the null rate under the
full grammar, the whole-run-planning thesis must be re-diagnosed before any
P1 planner is built. Expectation from the honest out-test (41.7% ante-8
had-out rate) and the 26.5% single-perturbation flip rate: 25-50%.

Run under the honest regime: BALATRO_NO_FORESIGHT=shuffle.

    PYTHONPATH=src python scripts/p11_route_oracle.py --seeds-file .data/honest_caps_300.jsonl \
        --max-seeds 100 --jobs 12 --out .data/p11_route_oracle.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

POLICIES = (
    "null",            # control: fork + unmodified bot (noise floor)
    "buy_best",        # every shop: buy best affordable joker offer (sell weakest if full)
    "hoard2_spend",    # skip purchases for 2 shops, then buy the 2 best affordable jokers
    "level_focus",     # every shop: buy planet/celestial offers + packs; skip jokers
    "reroll_hunt",     # every shop: reroll up to 2x before the bot's normal decisions
    "skip_smalls",     # skip every Small Blind from the fork on (tags over blind money)
    "churn_weakest",   # every shop: sell the weakest joker first, then default
    "buy_best_skip",   # buy_best + skip_smalls combined
)


def _shop_joker_offers(state):
    cards = state.modifiers.get("shop_cards", []) or []
    return [
        (i, sc) for i, sc in enumerate(cards)
        if isinstance(sc, dict) and str(sc.get("set", "")).upper() == "JOKER" and sc.get("name")
    ]


def _affordable(state, offer) -> bool:
    try:
        cost = int((offer.get("cost") or {}).get("buy", offer.get("price", 10**9)))
    except (TypeError, ValueError):
        return False
    return cost <= int(state.money)


class _RoutePolicyBot:
    """Wraps the deployed bot; overrides SHOP / BLIND_SELECT decisions per the
    route recipe, delegates everything else (plays stay basic-strategy)."""

    def __init__(self, policy: str, make_bot):
        from balatro_ai.api.actions import Action, ActionType
        from balatro_ai.api.state import GamePhase

        self._A, self._AT, self._GP = Action, ActionType, GamePhase
        self.policy = policy
        self.bot = make_bot()
        self.shop_key = None        # (ante, shop fingerprint) -> per-shop memory
        self.shop_seen = 0          # distinct shops visited since fork
        self.rerolls_this_shop = 0
        self.bought_this_shop = 0
        self.sold_this_shop = False

    def _new_shop(self, state) -> None:
        key = (int(state.ante), len(state.jokers), int(state.money))
        if self.shop_key != (int(state.ante),):
            self.shop_key = (int(state.ante),)
            self.shop_seen += 1
            self.rerolls_this_shop = 0
            self.bought_this_shop = 0
            self.sold_this_shop = False
        _ = key

    def _legal(self, state, action_type, **match):
        for a in state.legal_actions:
            if a.action_type != action_type:
                continue
            if all(a.metadata.get(k) == v for k, v in match.items()):
                return a
        return None

    def _buy_best_offer(self, state):
        offers = [(i, sc) for i, sc in _shop_joker_offers(state) if _affordable(state, sc)]
        if not offers:
            return None
        slot_limit = int(state.modifiers.get("joker_slot_limit", 5) or 5)
        if len(state.jokers) >= slot_limit:
            if self.sold_this_shop:
                return None
            weakest = min(
                range(len(state.jokers)),
                key=lambda j: int(state.jokers[j].sell_value or 0),
                default=None,
            )
            if weakest is None:
                return None
            self.sold_this_shop = True
            return self._A(self._AT.SELL, target_id="joker", amount=weakest,
                           metadata={"kind": "joker", "index": weakest})
        # priciest affordable offer as the value proxy
        i, _sc = max(offers, key=lambda o: int((o[1].get("cost") or {}).get("buy", 0) or 0))
        self.bought_this_shop += 1
        return self._A(self._AT.BUY, target_id="card", amount=i,
                       metadata={"kind": "card", "index": i})

    def choose_action(self, state):
        AT, GP = self._AT, self._GP
        p = self.policy
        if p == "null":
            return self.bot.choose_action(state)

        if state.phase == GP.BLIND_SELECT and p in ("skip_smalls", "buy_best_skip"):
            blind = str((state.modifiers.get("current_blind") or {}).get("name", state.blind or ""))
            if "Small" in blind:
                skip = next((a for a in state.legal_actions if a.action_type == AT.SKIP_BLIND), None)
                if skip is not None:
                    return skip

        if state.phase == GP.SHOP:
            self._new_shop(state)
            if p == "reroll_hunt" and self.rerolls_this_shop < 2:
                rr = next((a for a in state.legal_actions
                           if a.action_type == AT.REROLL
                           and str(a.metadata.get("kind", "")) != "boss"), None)
                if rr is not None:
                    self.rerolls_this_shop += 1
                    return rr
            if p in ("buy_best", "buy_best_skip") and self.bought_this_shop < 1:
                action = self._buy_best_offer(state)
                if action is not None:
                    return action
            if p == "hoard2_spend":
                if self.shop_seen <= 2:
                    end = next((a for a in state.legal_actions if a.action_type == AT.END_SHOP), None)
                    if end is not None:
                        return end
                elif self.bought_this_shop < 2:
                    action = self._buy_best_offer(state)
                    if action is not None:
                        return action
            if p == "level_focus":
                cards = state.modifiers.get("shop_cards", []) or []
                for i, sc in enumerate(cards):
                    if not isinstance(sc, dict) or not _affordable(state, sc):
                        continue
                    if str(sc.get("set", "")).upper() in ("PLANET", "CELESTIAL") and self.bought_this_shop < 2:
                        self.bought_this_shop += 1
                        return self._A(AT.BUY, target_id="card", amount=i,
                                       metadata={"kind": "card", "index": i})
            if p == "churn_weakest" and not self.sold_this_shop and len(state.jokers) >= 2:
                weakest = min(range(len(state.jokers)),
                              key=lambda j: int(state.jokers[j].sell_value or 0))
                self.sold_this_shop = True
                return self._A(AT.SELL, target_id="joker", amount=weakest,
                               metadata={"kind": "joker", "index": weakest})

        return self.bot.choose_action(state)


def _worker(arg):
    seed, fork_back = arg
    from dataclasses import replace as dcr
    from functools import partial

    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import GamePhase
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    cfg_scope = partial(bot_config_scope, dcr(DEFAULT_CONFIG, shop_audit_enabled=False))

    def make_bot():
        return create_bot("solver_shop_basic_play_bot", seed=0)

    def run_to_end(sim, bot):
        with cfg_scope():
            for _ in range(6000):
                st = sim.state
                if st.run_over:
                    break
                try:
                    a = bot.choose_action(st)
                except Exception:  # noqa: BLE001
                    break
                if a is None or a.action_type == ActionType.NO_OP:
                    break
                try:
                    sim.step(a)
                except Exception:  # noqa: BLE001
                    break
        return bool(sim.state.won), int(sim.state.ante), bool(getattr(sim, "_rng_diverged", True))

    # baseline (faithful) + rolling window of recent shop forks
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white", balatro_seed=seed)
    sim.state = SeedGame(seed, stake="white").initial_state()
    bot = make_bot()
    forks: list[tuple[int, object]] = []  # (ante, deepcopy) — keep last `fork_back`+1 antes
    seen = set()
    with cfg_scope():
        for _ in range(6000):
            st = sim.state
            if st.run_over:
                break
            if st.phase == GamePhase.SHOP and int(st.ante) not in seen:
                seen.add(int(st.ante))
                forks.append((int(st.ante), copy.deepcopy(sim)))
                forks = forks[-(fork_back + 1):]
            try:
                a = bot.choose_action(st)
            except Exception:  # noqa: BLE001
                break
            if a is None or a.action_type == ActionType.NO_OP:
                break
            sim.step(a)
    if bool(sim.state.won):
        return {"seed": seed, "base_won": True}
    death_ante = int(sim.state.ante)
    use_forks = [(a, s) for a, s in forks if a < death_ante][-fork_back:]
    if not use_forks:
        return {"seed": seed, "base_won": False, "death_ante": death_ante, "no_fork": True}

    rollouts = []
    for fork_ante, fsim in use_forks:
        for policy in POLICIES:
            g = copy.deepcopy(fsim)
            won, final_ante, diverged = run_to_end(g, _RoutePolicyBot(policy, make_bot))
            rollouts.append(
                {"fork_ante": fork_ante, "policy": policy, "won": won,
                 "final_ante": final_ante, "clean": not diverged}
            )
    return {"seed": seed, "base_won": False, "death_ante": death_ante, "rollouts": rollouts}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-file", required=True, help="JSONL with {'seed':...} rows (honest caps)")
    ap.add_argument("--max-seeds", type=int, default=100)
    ap.add_argument("--fork-back", type=int, default=2, help="fork at the last N pre-death shop antes")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seeds = []
    for line in open(args.seeds_file, encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            if not row.get("won"):
                seeds.append(row["seed"])
        if len(seeds) >= args.max_seeds:
            break
    print(f"[p11] {len(seeds)} honest-loss seeds, {len(POLICIES)} policies, fork_back={args.fork_back}", flush=True)

    tasks = [(s, args.fork_back) for s in seeds]
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(_worker, tasks))
    else:
        results = [_worker(t) for t in tasks]

    losses = [r for r in results if not r.get("base_won") and not r.get("no_fork")]
    null_flips = sum(
        1 for r in losses
        if any(x["won"] for x in r["rollouts"] if x["policy"] == "null" and x["clean"])
    )
    routed = sum(
        1 for r in losses
        if any(x["won"] for x in r["rollouts"] if x["policy"] != "null" and x["clean"])
    )
    policy_wins: Counter[str] = Counter()
    for r in losses:
        for x in r["rollouts"]:
            if x["won"] and x["clean"] and x["policy"] != "null":
                policy_wins[x["policy"]] += 1
    clean_frac = (
        sum(1 for r in losses for x in r["rollouts"] if x["clean"])
        / max(1, sum(len(r["rollouts"]) for r in losses))
    )
    summary = {
        "n_losses": len(losses),
        "null_flip_rate": round(null_flips / max(1, len(losses)), 3),
        "routed_rate": round(routed / max(1, len(losses)), 3),
        "routed_above_null": round((routed - null_flips) / max(1, len(losses)), 3),
        "clean_rollout_frac": round(clean_frac, 3),
        "policy_win_counts": dict(policy_wins.most_common()),
        "rows": results,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    print(f"[p11] written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
