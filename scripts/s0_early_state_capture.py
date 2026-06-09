"""S0 step 2a: capture EARLY-STATE features on the baseline trajectory per seed.

For each seed, run the oracle's exact baseline policy (deployed SolverPolicy shop +
BasicStrategy play, archetype=None) and snapshot the GameState at the ante-2 and ante-3
blind-select via a policy wrapper (so the captured state is exactly the baseline
trajectory the oracle scored). Extracts the hypothesized early signals -- especially
DECK SUIT CONCENTRATION (the never-tried early-flush detector) -- plus hand levels,
owned/seen archetype key-jokers, money. Joined later with oracle best-basin labels to
train the predictive-foresight classifier (the S0 gate).

    PYTHONPATH=src py -3.12 scripts/s0_early_state_capture.py --seeds 200 --seed-offset 1000000 \
        --jobs 8 --stake white --out .data/s0_early_features_white_200.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CAPTURE_ANTES = (2, 3)
SUITS = ("S", "H", "D", "C")
HAND_TYPES = ("Flush", "Straight", "Two Pair", "Pair", "Three of a Kind",
              "High Card", "Four of a Kind", "Straight Flush", "Full House")


def _is_stone(card) -> bool:
    enh = getattr(card, "enhancement", None)
    return bool(enh) and "stone" in str(enh).lower()


def _deck_suit_features(state) -> dict:
    counts = Counter()
    for c in getattr(state, "known_deck", ()) or ():
        if _is_stone(c):
            continue
        s = getattr(c, "suit", None)
        if s in SUITS:
            counts[s] += 1
    total = sum(counts.values()) or 1
    fr = {s: counts.get(s, 0) / total for s in SUITS}
    red = (counts.get("H", 0) + counts.get("D", 0)) / total
    black = (counts.get("S", 0) + counts.get("C", 0)) / total
    feats = {f"suit_frac_{s}": round(fr[s], 4) for s in SUITS}
    feats["suit_max_single"] = round(max(fr.values()), 4)
    feats["suit_max_smeared_pair"] = round(max(red, black), 4)
    feats["deck_nonstone"] = total
    return feats


def _hand_level_features(state) -> dict:
    hl = getattr(state, "hand_levels", {}) or {}
    out = {}
    for ht in ("Flush", "Straight", "Pair", "Two Pair", "Three of a Kind", "High Card"):
        try:
            out[f"lvl_{ht.replace(' ', '_').lower()}"] = int(hl.get(ht, 1) or 1)
        except (TypeError, ValueError):
            out[f"lvl_{ht.replace(' ', '_').lower()}"] = 1
    return out


def _worker(arg):
    seed, stake = arg
    from balatro_ai.api.actions import Action, ActionType
    from balatro_ai.api.state import GamePhase
    from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.solver.archetypes import BUILT_IN_ARCHETYPES, _joker_key
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.trajectory import generate_trajectory
    from dataclasses import replace as dcr

    fb = BasicStrategyBot(seed=0)
    policy = SolverPolicy(play_policy=fb, fallback=fb, seed=0,
                          prefer_fallback_info_first_shop=True, fallback_negative_shop_sells=True)

    arch_keys = {a.name: a.key_joker_keys for a in BUILT_IN_ARCHETYPES}
    seen = {a.name: 0 for a in BUILT_IN_ARCHETYPES}
    snaps: dict[int, dict] = {}
    done_ante = max(CAPTURE_ANTES)

    def _owned_seen_features(state):
        f = {}
        for name, keys in arch_keys.items():
            owned = sum(1 for j in state.jokers if _joker_key(j) in keys)
            f[f"owned_key_{name}"] = owned
            f[f"seen_key_{name}"] = seen[name]
        return f

    def wrapped(state):
        # accumulate archetype key-jokers seen in shops
        if state.phase == GamePhase.SHOP:
            for sc in (state.modifiers.get("shop_cards") or []):
                if not isinstance(sc, dict):
                    continue
                if str(sc.get("set", "")).upper() != "JOKER":
                    continue
                k = sc.get("key") or sc.get("card_key")
                if not (isinstance(k, str) and k.startswith("j_")):
                    continue
                for name, keys in arch_keys.items():
                    if k in keys:
                        seen[name] += 1
        # snapshot at the ante-2 / ante-3 blind-select (before committing anything)
        if state.phase == GamePhase.BLIND_SELECT and int(state.ante) in CAPTURE_ANTES \
                and int(state.ante) not in snaps:
            a = int(state.ante)
            feats = {"ante": a, "money": int(state.money),
                     "n_jokers": len(state.jokers), "deck_size": int(state.deck_size)}
            feats.update(_deck_suit_features(state))
            feats.update(_hand_level_features(state))
            feats.update(_owned_seen_features(state))
            snaps[a] = feats
            # once we've captured the deepest ante, end the run early to save compute
            if all(x in snaps for x in CAPTURE_ANTES) or a >= done_ante:
                return Action(ActionType.NO_OP)
        return policy.choose_action(state)

    with bot_config_scope(dcr(DEFAULT_CONFIG, shop_audit_enabled=False)):
        generate_trajectory(seed, wrapped, stake=stake, max_steps=5000, record_steps=False)
    return {"seed": seed, "snaps": snaps}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--seed-offset", type=int, default=1000000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--stake", default="white")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds = [f"{args.seed_offset + i:07d}" for i in range(1, args.seeds + 1)]
    print(f"[s0-capture] {len(seeds)} seeds, baseline early-state at antes {CAPTURE_ANTES}, stake={args.stake}", flush=True)

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = list(ex.map(_worker, [(s, args.stake) for s in seeds]))
    else:
        rows = [_worker((s, args.stake)) for s in seeds]

    got2 = sum(1 for r in rows if 2 in r["snaps"])
    got3 = sum(1 for r in rows if 3 in r["snaps"])
    print(f"captured ante2 for {got2}/{len(rows)}, ante3 for {got3}/{len(rows)}", flush=True)
    feats = {r["seed"]: {str(a): r["snaps"][a] for a in r["snaps"]} for r in rows}
    json.dump(feats, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
