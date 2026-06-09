"""Trace shop decisions to hunt a SYSTEMATIC build mis-valuation.

The ec9d0b7 win came from finding a systematic PLAY-value error (cleared blind
valued below almost-cleared) by tracing per-candidate values. This is the shop
analogue: for each shop phase in a game, dump the leaf-term breakdown of the
state AFTER each legal shop action (buy each item, reroll, end shop), so we can
see what the value function THINKS of each option and spot a systematic error
(e.g. END_SHOP out-valuing buying a strong xmult joker, money over-valued,
build/role gains under-valued). Prints the chosen action for comparison.

    PYTHONPATH=src python scripts/shop_decision_trace.py [seed] [max_shops]
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", nargs="?", default="0000020")
    parser.add_argument("max_shops", nargs="?", type=int, default=6)
    parser.add_argument("start_shop", nargs="?", type=int, default=0)
    parser.add_argument("--bot", default="solver_shop_basic_play_bot")
    args = parser.parse_args()

    import balatro_ai.search.hand_search as hs
    from balatro_ai.api.actions import ActionType
    from balatro_ai.bots.config import bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.search.shop_search import shop_leaf_terms
    from balatro_ai.search.forward_sim import simulate_buy, simulate_sell, simulate_end_shop
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    hs._NATIVE_BEAM_ENABLED = False
    sim = LocalBalatroSimulator(seed=_stable_seed_int(args.seed), stake="white")
    sim.state = SeedGame(args.seed, stake="white").initial_state()
    bot = create_bot(args.bot, seed=0)

    shop_types = {ActionType.BUY, ActionType.SELL, ActionType.REROLL,
                  ActionType.OPEN_PACK, ActionType.END_SHOP}
    total_shop_steps = 0
    for _ in range(6000):
        st = sim.state
        if st.run_over:
            break
        phase = str(st.phase.value)
        legal = [a for a in st.legal_actions if a.action_type in shop_types] if phase == "shop" else []
        in_range = bool(legal) and (args.start_shop <= total_shop_steps < args.start_shop + args.max_shops)
        if legal:
            total_shop_steps += 1
            if in_range:
                with bot_config_scope(_bot_config(bot)):
                    jk = ", ".join(getattr(j, "name", "?") for j in st.jokers)
                    root_terms = shop_leaf_terms(st, root_state=st)
                    print(f"\n=== SHOP step #{total_shop_steps}  ante={st.ante} money=${st.money} "
                          f"jokers=[{jk}] ===")
                    print(f"  current-state total={root_terms.total:.2f}  "
                          f"levels={ {k:v for k,v in (st.hand_levels or {}).items() if v>1} }")
                    rows = []
                    for a in legal:
                        try:
                            if a.action_type == ActionType.BUY:
                                ns = simulate_buy(st, a)
                            elif a.action_type == ActionType.SELL:
                                ns = simulate_sell(st, a)
                            elif a.action_type == ActionType.END_SHOP:
                                ns = simulate_end_shop(st)
                            else:
                                continue  # skip reroll (needs RNG-faithful cards)
                        except Exception as e:  # noqa: BLE001 - diagnostic
                            rows.append((a, None, f"err:{type(e).__name__}"))
                            continue
                        t = shop_leaf_terms(ns, root_state=st, root_build_score=root_terms.build_score)
                        rows.append((a, t, ""))
                    rows.sort(key=lambda r: (r[1].total if r[1] else -1e9), reverse=True)
                    for a, t, err in rows[:12]:
                        label = _action_label(a, st)
                        if t is None:
                            print(f"    {label:34s} {err}")
                            continue
                        d = t.to_trace_dict()
                        print(f"    {label:34s} total={t.total:7.2f}  "
                              f"role={d['roles']:5.1f} build={d['build_delta']:6.2f} "
                              f"owned={d['owned']:6.2f} money={d['money']:5.1f} "
                              f"lvl={d.get('leveling',0):5.1f} coh={d.get('coherence',0):5.1f} "
                              f"surv={d['survival']:5.1f}")
                    _dump_beam_candidates(st, bot)

        a = bot.choose_action(st)
        if a.action_type.value == "no_op":
            break
        if in_range:
            print(f"  >>> CHOSEN: {_action_label(a, st)}")
        sim.step(a)

    s = sim.state
    print(f"\nfinal: ante={s.ante} score={int(s.current_score)}")
    return 0


def _bot_config(bot):
    return getattr(bot, "config", None)


def _dump_beam_candidates(st, bot) -> None:
    """Dump the ACTUAL beam's top candidate paths + scores for this decision."""
    from dataclasses import replace as dc_replace
    from balatro_ai.search.shop_search import best_shop_action, ShopSearchContext
    pol = getattr(bot, "_policy", bot)
    if not hasattr(pol, "shop_config") or not hasattr(pol, "_sampler"):
        return
    cfg = dc_replace(pol.shop_config, trace_top_paths=10)
    try:
        act = best_shop_action(st, config=cfg, sampler=pol._sampler,
                               protected_jokers=getattr(pol, "_protected_shop_jokers", ()),
                               shop_context=ShopSearchContext(
                                   rerolls_in_shop=getattr(pol, "_rerolls_in_shop", 0),
                                   packs_opened_in_shop=getattr(pol, "_packs_opened_in_shop", 0),
                                   filled_last_joker_slot=getattr(pol, "_filled_last_joker_slot_in_shop", False),
                               ))
    except Exception as e:  # noqa: BLE001 - diagnostic
        print(f"    [beam probe err: {type(e).__name__}: {e}]")
        return
    cands = (act.metadata or {}).get("search_candidates", ()) if act else ()
    print("    --- BEAM top paths (score = action_score + leaf*0.35) ---")
    for c in cands:
        path = " -> ".join(c.get("path_labels") or [_short(p) for p in c.get("path", ())])
        print(f"      score={c['score']:8.2f}  act={c['action_score']:8.2f} "
              f"leaf={c['leaf_score']:8.2f}  [{path}]  -> jokers={c['result']['jokers']}")


def _short(p) -> str:
    if isinstance(p, dict):
        return str(p.get("action_type", p))
    return str(p)


def _action_label(a, st) -> str:
    t = a.action_type.value
    meta = a.metadata or {}
    kind = meta.get("kind", "")
    idx = meta.get("index", None)
    name = ""
    if kind == "card" and idx is not None:
        shop = getattr(st, "shop", None) or ()
        if isinstance(shop, (tuple, list)) and 0 <= idx < len(shop):
            name = str(shop[idx])
    elif t == "sell" and idx is not None:
        jokers = getattr(st, "jokers", None) or ()
        if 0 <= idx < len(jokers):
            name = "SELL " + str(getattr(jokers[idx], "name", "?"))
    suffix = f"[{idx}]" if idx is not None else ""
    return f"{t}{suffix} {name}".strip()


if __name__ == "__main__":
    raise SystemExit(main())
