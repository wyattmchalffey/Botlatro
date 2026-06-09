"""S-pre Part 1: classify the 226 ante-8 out jokers as compounder vs additive/decay/retrigger.
Distinguishes thesis A (scaling basin) from B (additive cushion / economy)."""
import json
from collections import Counter
import balatro_ai.bots.basic_strategy.data as d

SCALING = set(d.SCALING_JOKERS); XMULT = set(d.XMULT_JOKERS)
ECON = set(getattr(d, "JOKER_ECONOMY_VALUES", {}))
# decay / one-time / temporary: in a mult/chip/xmult set but does NOT compound
DECAY = {"Gros Michel", "Ice Cream", "Popcorn", "Ramen", "Turtle Bean", "Luchador",
         "Diet Cola", "Invisible Joker", "Hallucination", "Mr. Bones", "Egg"}
RETRIGGER = {"Hanging Chad", "Mime", "Sock and Buskin", "Dusk", "Hack", "Seltzer",
             "Hanging Chad Joker"}
UTILITY = {"Splash", "Smeared Joker", "Four Fingers", "Shortcut", "Pareidolia",
           "Credit Card", "Chaos the Clown", "Showman", "DNA", "Marble Joker"}

def classify(name):
    if name in RETRIGGER: return "retrigger"
    if name in DECAY: return "decay/onetime"
    if name in UTILITY: return "utility"
    if (name in SCALING or name in XMULT): return "compounder"
    if name in ECON: return "economy"
    return "additive/flat"

d2 = json.load(open(".data/endgame_out_test.json"))
outs = []
for r in d2.get("rows", []):
    outs += r.get("outs", [])
cls = Counter(classify(n) for n in outs)
n = len(outs)
print(f"=== S-pre Part 1: {n} ante-8 'out' jokers across {d2['n_with_out']} losses-with-out ===")
for k, v in cls.most_common():
    print(f"  {k:16s}: {v:3d} ({v/n:.1%})")
comp = cls["compounder"] + cls["retrigger"]
print(f"\n  COMPOUNDER+RETRIGGER (thesis A): {comp} ({comp/n:.1%})")
adde = cls["additive/flat"] + cls["decay/onetime"] + cls["economy"]
print(f"  ADDITIVE/DECAY/ECON (thesis B):  {adde} ({adde/n:.1%})")
print(f"  utility:                        {cls['utility']} ({cls['utility']/n:.1%})")
# also: per-loss, did it have a compounder out at all?
has_comp = sum(1 for r in d2["rows"] if any(classify(x) in ("compounder","retrigger") for x in r.get("outs",[])))
nrows = sum(1 for r in d2["rows"] if r.get("outs"))
print(f"\n  losses-with-out that had >=1 compounder/retrigger out: {has_comp}/{nrows} ({has_comp/max(1,nrows):.1%})")
