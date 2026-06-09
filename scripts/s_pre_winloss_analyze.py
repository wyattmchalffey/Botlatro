"""S-pre Part 2 analysis: winners vs losers, matched on reaching ante a.

Effect size (Cohen's d) for each feature at antes 4/5/6 distinguishes the lever:
  - big d on `compounder` -> thesis A (scaling basin: winners own growth engines losers don't)
  - big d on `max_hand_level`/`n_types_leveled`(low)/`money` -> thesis B (concentration + economy)
Also reports build-capacity growth multiplier per ante (the convex-growth signature).

    PYTHONPATH=src py -3.12 scripts/s_pre_winloss_analyze.py --cap .data/s_pre_winloss_200.json
"""
from __future__ import annotations
import argparse, json, statistics as st

FEATURES = ["compounder", "retrigger", "additive", "decay", "economy",
            "max_hand_level", "n_types_leveled", "sum_levels", "money", "build_score"]


def cohend(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.pvariance(a), st.pvariance(b)
    pooled = (((len(a)-1)*va + (len(b)-1)*vb) / max(1, len(a)+len(b)-2)) ** 0.5
    return (ma - mb) / pooled if pooled > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True)
    args = ap.parse_args()
    rows = json.load(open(args.cap, encoding="utf-8"))
    n = len(rows)
    wins = [r for r in rows if r["won"]]
    print(f"=== S-pre Part 2: {n} runs, winrate {len(wins)}/{n} ({len(wins)/n:.1%}) ===")
    print(f"final-ante dist: " + ", ".join(f"{a}:{sum(1 for r in rows if r['final_ante']==a)}" for a in range(1, 9)))

    def snap(r, a):
        return r["snaps"].get(str(a))

    for a in (3, 4, 5, 6, 7):
        # matched: runs that REACHED ante a (have a snapshot), split by eventual win
        reached = [r for r in rows if snap(r, a)]
        w = [r for r in reached if r["won"]]
        l = [r for r in reached if not r["won"]]
        if len(w) < 3 or len(l) < 3:
            print(f"\n-- ante {a}: too few (w={len(w)} l={len(l)}) --")
            continue
        print(f"\n-- ante {a}: reached by {len(reached)} (winners {len(w)}, losers {len(l)}) --")
        print(f"   {'feature':16s} {'winners':>9s} {'losers':>9s} {'Cohen_d':>8s}")
        scored = []
        for f in FEATURES:
            wv = [snap(r, a)[f] for r in w]
            lv = [snap(r, a)[f] for r in l]
            d = cohend(wv, lv)
            scored.append((abs(d) if d == d else 0, f, st.mean(wv), st.mean(lv), d))
        for absd, f, mw, ml, d in sorted(scored, reverse=True):
            print(f"   {f:16s} {mw:9.2f} {ml:9.2f} {d:+8.2f}")

    # build-capacity growth multiplier per ante (winners vs losers, where both antes present)
    print("\n=== build_score growth multiplier (ante a -> a+1), winners vs losers ===")
    for a in (3, 4, 5, 6, 7):
        def mults(rs):
            out = []
            for r in rs:
                s0, s1 = snap(r, a), snap(r, a + 1)
                if s0 and s1 and s0["build_score"] > 1:
                    out.append(s1["build_score"] / s0["build_score"])
            return out
        wm, lm = mults(wins), mults([r for r in rows if not r["won"]])
        sw = f"{st.median(wm):.2f}x (n{len(wm)})" if wm else "-"
        sl = f"{st.median(lm):.2f}x (n{len(lm)})" if lm else "-"
        print(f"   ante {a}->{a+1}: winners {sw}   losers {sl}")


if __name__ == "__main__":
    raise SystemExit(main())
