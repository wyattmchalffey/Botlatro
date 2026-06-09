"""S0 kill-switch: does forcing a DIFFERENT buy at an ante-3/4/5 shop flip a loss to a win?

For each deployed-bot loss seed: replay, fork the sim at each ante 3/4/5 shop, and for every
affordable offered joker, force-buy it then let the bot play the rest of the run to terminal. An
"out" = a forced mid-game buy that flips the run from loss to win. This tests the load-bearing
thesis of the from-scratch core: that the bot's losses are mid-game build-CONSTRUCTION failures
recoverable by a different buy SEQUENCE (here, a single forced buy with full runway), not RNG.

Validity note: the buy is at ante <=5 (RNG-validated tape); the roll-forward through antes 6-8 uses
the sampler tail (unvalidated) -> outcomes are directional, not exact. Reported by buy-ante and
joker type (compounder vs additive) per S-pre.

    PYTHONPATH=src py -3.12 scripts/s0_midgame_construction.py --seeds 120 --seed-offset 5000000 \
        --jobs 8 --antes 3,4,5 --out .data/s0_midgame.json
"""
from __future__ import annotations
import argparse, copy, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DECAY = {"Gros Michel", "Ice Cream", "Popcorn", "Ramen", "Turtle Bean", "Luchador",
         "Diet Cola", "Invisible Joker", "Hallucination", "Mr. Bones", "Egg"}
RETRIGGER = {"Hanging Chad", "Mime", "Sock and Buskin", "Dusk", "Hack", "Seltzer"}


def _run_to_end(sim, bot, cfg_scope, max_steps=4000):
    from balatro_ai.api.actions import ActionType
    with cfg_scope():
        for _ in range(max_steps):
            st = sim.state
            if st.run_over:
                break
            try:
                a = bot.choose_action(st)
            except Exception:
                break
            if a is None or a.action_type == ActionType.NO_OP:
                break
            try:
                sim.step(a)
            except Exception:
                break
    # diverged = the run fell back from seed-faithful keyed RNG to the sequential
    # sampler at some point (so future shops became action-dependent -> contaminated).
    diverged = bool(getattr(sim, "_rng_diverged", True))
    return bool(sim.state.won), int(sim.state.ante), diverged


def _worker(arg):
    seed, antes = arg
    from functools import partial
    from dataclasses import replace as dcr
    from balatro_ai.api.actions import Action, ActionType
    from balatro_ai.api.state import GamePhase
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    import balatro_ai.bots.basic_strategy.data as d

    SCALING = set(d.SCALING_JOKERS); XMULT = set(d.XMULT_JOKERS)

    def jtype(name):
        if name in RETRIGGER: return "retrigger"
        if name in DECAY: return "decay"
        if name in SCALING or name in XMULT: return "compounder"
        return "additive"

    cfg_scope = partial(bot_config_scope, dcr(DEFAULT_CONFIG, shop_audit_enabled=False))

    def fresh_bot():
        return create_bot("solver_shop_basic_play_bot", seed=0)

    # 1) baseline run (FAITHFUL mode: balatro_seed -> keyed, action-independent shop prediction,
    # so a forced buy does NOT spuriously shift future shops). Record forks at the target shops.
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white", balatro_seed=seed)
    sim.state = SeedGame(seed, stake="white").initial_state()
    bot = fresh_bot()
    forks = []  # (ante, deepcopy(sim))
    seen_ante = set()
    with cfg_scope():
        for _ in range(6000):
            st = sim.state
            if st.run_over:
                break
            if st.phase == GamePhase.SHOP and int(st.ante) in antes and int(st.ante) not in seen_ante:
                seen_ante.add(int(st.ante))
                forks.append((int(st.ante), copy.deepcopy(sim)))
            try:
                a = bot.choose_action(st)
            except Exception:
                break
            if a.action_type == ActionType.NO_OP:
                break
            sim.step(a)
    base_won = bool(sim.state.won); base_ante = int(sim.state.ante)
    base_diverged = bool(getattr(sim, "_rng_diverged", True))
    if base_won:
        return {"seed": seed, "base_won": True, "base_ante": base_ante, "outs": [],
                "base_diverged": base_diverged}

    from balatro_ai.api.state import with_derived_legal_actions
    # CONTROLS: null (continue, no intervention) + reroll (perturb shop, no forced build gain).
    # Each rollout records whether it stayed seed-faithful (clean) or fell back to the sampler.
    null_won = False
    reroll = []  # (won, clean)
    for ante, fsim in forks:
        gc = copy.deepcopy(fsim)
        w, _, dv = _run_to_end(gc, fresh_bot(), cfg_scope)
        if w:
            null_won = True
        st2 = with_derived_legal_actions(fsim.state)
        rr = next((a for a in st2.legal_actions if a.action_type == ActionType.REROLL
                   and str(a.metadata.get("kind", "")) != "boss"), None)
        if rr is not None:
            gr = copy.deepcopy(fsim)
            try:
                gr.step(rr)
                w, _, dv = _run_to_end(gr, fresh_bot(), cfg_scope)
                reroll.append((bool(w), not dv))
            except Exception:
                pass

    # INTERVENTIONS: force-buy/swap each affordable offered joker; record (won, clean) per rollout.
    outs = []
    inter = []  # (won, clean)
    tried = set()
    for ante, fsim in forks:
        st = fsim.state
        cards = st.modifiers.get("shop_cards", []) or []
        slot_limit = int(st.modifiers.get("joker_slot_limit", 5) or 5)
        open_slot = len(st.jokers) < slot_limit
        joffers = [(i, sc.get("name")) for i, sc in enumerate(cards)
                   if isinstance(sc, dict) and str(sc.get("set", "")).upper() == "JOKER" and sc.get("name")]
        for i, name in joffers:
            if name in tried:
                continue
            sell_opts = [None] if open_slot else list(range(len(st.jokers)))
            won_here = False
            for O in sell_opts:
                g = copy.deepcopy(fsim)
                try:
                    if O is not None:
                        g.step(Action(ActionType.SELL, target_id="joker", amount=O,
                                      metadata={"kind": "joker", "index": O}))
                    nb = len(g.state.jokers)
                    g.step(Action(ActionType.BUY, target_id="card", amount=i,
                                  metadata={"kind": "card", "index": i}))
                except Exception:
                    continue
                if len(g.state.jokers) <= nb:
                    continue
                won, _, dv = _run_to_end(g, fresh_bot(), cfg_scope)
                inter.append((bool(won), not dv))
                if won:
                    outs.append({"ante": ante, "joker": name, "type": jtype(name),
                                 "swap": O is not None, "clean": not dv})
                    won_here = True
                    break
            if won_here:
                tried.add(name)
    return {"seed": seed, "base_won": False, "base_ante": base_ante, "outs": outs,
            "n_rollouts": len(inter), "n_forks": len(forks), "base_diverged": base_diverged,
            "null_won": null_won, "reroll": reroll, "inter": inter,
            "reroll_won": any(w for w, _ in reroll)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=120)
    ap.add_argument("--seed-offset", type=int, default=5000000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--antes", default="3,4,5")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    antes = frozenset(int(x) for x in args.antes.split(","))
    seeds = [f"{args.seed_offset + i:07d}" for i in range(1, args.seeds + 1)]
    print(f"[s0-midgame] {len(seeds)} seeds, force-buy at antes {sorted(antes)}, roll to terminal", flush=True)
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = list(ex.map(_worker, [(s, antes) for s in seeds]))
    else:
        rows = [_worker((s, antes)) for s in seeds]

    from collections import Counter
    base_wins = sum(r["base_won"] for r in rows)
    base_div = sum(1 for r in rows if r.get("base_diverged"))
    losses = [r for r in rows if not r["base_won"]]
    with_out = [r for r in losses if r["outs"]]
    null_w = sum(1 for r in losses if r.get("null_won"))
    # flatten (won, clean) rollout lists
    inter = [t for r in losses for t in r.get("inter", [])]
    reroll = [t for r in losses for t in r.get("reroll", [])]

    def rate(lst):  # (wins, attempts, per-attempt %)
        w = sum(1 for won, _ in lst if won)
        return w, len(lst), (w / len(lst) if lst else 0.0)

    iw, ia, ip = rate(inter)
    rw, ra, rp = rate(reroll)
    inter_clean = [t for t in inter if t[1]]
    reroll_clean = [t for t in reroll if t[1]]
    icw, ica, icp = rate(inter_clean)
    rcw, rca, rcp = rate(reroll_clean)
    print(f"\nbaseline winrate {base_wins}/{len(rows)} ({base_wins/len(rows):.1%}); losses {len(losses)}")
    print(f"baseline runs that DIVERGED (fell back to sampler): {base_div}/{len(rows)} ({base_div/len(rows):.1%})")
    print(f"CONTROL null-win: {null_w}/{len(losses)} ({null_w/max(1,len(losses)):.1%})  [should be ~0 if faithful]")
    print(f"clean (seed-faithful) rollout fraction: inter {ica}/{ia} ({ica/max(1,ia):.0%}), reroll {rca}/{ra} ({rca/max(1,ra):.0%})")
    print(f"\n=== PER-ATTEMPT win rate (the fair comparison) ===")
    print(f"  ALL rollouts:   intervention {iw}/{ia}={ip:.1%}   reroll {rw}/{ra}={rp:.1%}")
    print(f"  CLEAN only:     intervention {icw}/{ica}={icp:.1%}   reroll {rcw}/{rca}={rcp:.1%}   <- the contamination-free signal")
    print(f"  -> on the clean subset, intervention {'BEATS' if icp>rcp+0.02 else ('~= ' if abs(icp-rcp)<=0.02 else 'LOSES to')} reroll")
    print(f"\nlosses with >=1 intervention OUT (upper bound, multi-comparison-inflated): {len(with_out)}/{len(losses)} ({len(with_out)/max(1,len(losses)):.1%})")
    clean_out = [r for r in losses if any(o.get("clean") for o in r["outs"])]
    print(f"losses with a CLEAN intervention out: {len(clean_out)}/{len(losses)} ({len(clean_out)/max(1,len(losses)):.1%})")
    # by buy-ante and joker type
    ante_c = Counter(); type_c = Counter(); type_loss = Counter()
    for r in with_out:
        types = set()
        for o in r["outs"]:
            ante_c[o["ante"]] += 1
            type_c[o["type"]] += 1
            types.add(o["type"])
        for t in types:
            type_loss[t] += 1
    print(f"out instances by buy-ante: {dict(ante_c)}")
    print(f"out instances by joker type: {dict(type_c)}")
    print(f"losses-with-out that had a COMPOUNDER/RETRIGGER out: "
          f"{type_loss['compounder']+type_loss['retrigger']}/{len(with_out)}")
    print(f"losses-with-out that had ONLY additive/decay outs: "
          f"{sum(1 for r in with_out if all(o['type'] in ('additive','decay') for o in r['outs']))}/{len(with_out)}")
    json.dump(rows, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
