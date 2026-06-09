# New play+build core — design plan (2026-06-09)

> **S0 RESULT (2026-06-09): the kill-switch came back NEGATIVE — premise REFUTED (contamination-free).**
> S-pre verdict was thesis B > A (build power = a combination of mature heuristics; no cheap knob;
> decay A/B neutral). The S0 mid-game construction test (`s0_midgame_construction.py`, 120 seeds,
> fork at antes 3-5 + force every realizable buy/swap + roll to terminal). First run was GENERIC
> (sampler) mode -> contaminated. **Re-run in FAITHFUL mode (balatro_seed: keyed, action-independent
> shops) and partitioned each rollout by whether it stayed seed-faithful (`_rng_diverged`):**
> null-control 0.0% (faithful). **DEFINITIVE v2 run (`s0_midgame_faithful_v2.json`): after the two
> source-validated RNG-faithfulness fixes the clean subset is now ~100% (intervention 1187/1191,
> reroll 269/271 stayed seed-faithful), and on that contamination-free AND unbiased subset forced
> build interventions win 5.0%/attempt (59/1187) vs neutral rerolls 4.8%/attempt (13/269) — they are
> statistically indistinguishable.** (The earlier 1.7%-vs-5.6% gap was a bias artifact of the old
> ~61%-clean contaminated subset, NOT a real effect; with the RNG fixes the clean fraction jumped to
> ~100% and the gap vanished.) Only 27/102 losses (26.5%) had ANY clean seed-faithful perturbation
> that flips them, and a forced buy is no better than a blind reroll at finding it; ~73% of losses are
> unrecoverable mid-game. **Conclusion: the bot's mid-game build SELECTION is near-optimal; the core's
> premise (losses recoverable by better construction) is REFUTED — now contamination-free AND
> unbiased. ~22-26% is robustly this architecture's ceiling. Do NOT build the core.** Exceeding ~26%
> would require a fundamentally stronger whole-run POLICY (large-scale RL self-play + validated ante-8
> RNG) -- a multi-month research program with a now-robustly-discouraging prior (per-decision
> selection is already near-optimal). The reroll-vs-intervention parity also closes the lone faint
> hint: "reroll-more-when-behind" was A/B-tested and REGRESSED (21->14/128, lost 7 / gained 0) -- the
> bot's reroll/economy discipline is near-optimal too. EVERY testable lever is now closed.

Goal: a from-scratch play+build core that can break past the ~26% white-stake ceiling of the
current heuristic architecture. Produced from a 17-agent design workflow (5 independent core
designs, adversarial critiques, synthesis + completeness critic that re-verified every load-bearing
number against on-disk artifacts). **Honest headline: the realistic reachable target is ~30–40%
white — a real break past 26%, but NOT superhuman (~80%). Superhuman is foreclosed on this
architecture + 8-core compute (reasons in §6). ~45–55% chance even ~30–40% does not materialize.**

---

## 1. Why 26% is a basin, not a wall

The bot wins ~22–26% white; strong humans win ~80%+. The game has huge headroom. The bot's ceiling
is mechanical: at the shop it optimizes a **concave, next-boss, current-counter** quantity
(`shop_search.py`: the build-delta leaf `shop_build_capacity_delta_value` is clamped to ±95 at
line 573; the separate headroom term `min(42.0,…)` at line 614; the win-probability target is
weighted 0.0 before ante 7). A *fresh scaling joker* reads ~0 immediate delta, so the bot never buys
into it. Balatro is won by a **convex, whole-run, projected-counter** quantity. Every prior
"shop/play/value is near-optimal/flat" measurement was a *local derivative inside the bot's weak
basin* — you cannot measure the value of a basin you never enter. That is the gap the new core must
close: **enter the strong scaling/concentration basins humans use, by making the build-value
forward-looking and per-candidate.**

S0 (the foresight gate, 2026-06-09) ruled out *predicting* the basin from early state (AUC at
chance) — but it did NOT rule out *adaptively constructing* it as pieces appear. The new core never
predicts; it reacts.

## 2. Ground truth the workflow re-verified against disk (corrections to prior assumptions)

| Claim | Verified on disk | Consequence for the design |
|---|---|---|
| ante-8 out-test "outs" are scaling cores | **FALSE — 81% additive/decay, 12.4% compounders, 6.6% retrigger** (`endgame_out_test.json`, 226 outs) | The scaling-basin thesis has a **yellow flag**. EITHER the bot needs additive *cushion* (economy/buy-timing) OR it needs to construct compounders *earlier* than the terminal blind — which the out-test (single graft at the failing blind, zero runway) structurally cannot test. The first experiment must test mid-game construction. |
| endgame play is an untapped lever | **Mostly dead, but the number is a LOWER BOUND.** `endgame_play_audit.json`: a depth-6/width-6 *beam* (NOT a true oracle) clears only 10.5% of ante-8 losses; median ratio 0.493 < bot's 0.743. | Deaths are build-limited, not play-recoverable. BUT the "oracle" is a sub-oracle — 10.5% is a floor and the median-worse is beam leaf error, not proof play is intrinsically dead. **Play is a minor lever; do not over-invest, but do not claim it's fully dead.** The one worthwhile play change: aim play at the committed/leveled hand type. |
| archetype lever | **already flat.** Oracle +6.2% flush-degenerate; live selector 0.22→0.23. | Do not re-open archetype selection. |
| caps file enables free per-candidate labels | **FALSE** — `onpolicy_solver_caps_384.jsonl` holds only `[score,ante,money]` + shop *indices*; no joker identities, no hand_levels. | Any "free offline" first experiment is partly a mirage; per-candidate value needs an **instrumented re-capture** (1 day of 8-core gen) and sim re-forks. |
| clearcap build-strength is a 0.90-AUC signal | v1 AUC 0.904 overall, but **v2 deep-ante (5–8) AUC ≈ 0.62–0.73** | The build-value's *realistic operating signal at the antes that matter* is ~0.65–0.73, not 0.90. Plan to the weaker number. |
| winners show 1.4–1.5×/ante late growth | true **but measured on the older ~17.7% caps**, not the deployed bot | Re-derive the growth/concentration signature on a deployed-bot capture before treating it as the target. |

## 3. The two live theses (the first experiments must distinguish them)

The 81%-additive-outs + winner-hand-concentration data make this genuinely open:

- **Thesis A — scaling basin.** Losses are recoverable by a different *sequence* of buys at antes 3–6
  that constructs a compounding engine (xmult/retrigger anchor + concentrated leveling). The myopic
  leaf misprices it. Fix = a constructive trajectory build-value.
- **Thesis B — concentration + economy.** Losses are recoverable by *additive* play: concentrate
  leveling on ONE hand type, bank interest to the $25 cap earlier, spend out late for cushion.
  Supported by the 81% additive outs + winners' 59% Pair/HighCard concentration + lower discard rate.

These are not mutually exclusive, but they imply different cores. **The first experiments below are
designed to tell which is real before committing build effort.**

### S-pre VERDICT (run 2026-06-09): thesis B > thesis A. Plus a concrete near-term lever.

Part 1 (226 terminal-blind outs): **72% additive/decay, 20% compounder/retrigger** — but 62% of
losses had a compounder available (terminal-graft is additive-biased by construction; inconclusive
on A). Part 2 (200 deployed runs, 20.5% win, winners-vs-losers Cohen's d at matched antes):

| Separator (ante 5 → ante 7 Cohen's d) | Reading |
|---|---|
| **build_score (capacity): +0.49 → +0.89, widening** | The outcome to explain: winners' builds are far stronger, gap grows late. |
| **money: +0.51 → +0.26** (strongest early) | Economy is a top separator (partly confounded by winning, but large). **Thesis B.** |
| **decay jokers: −0.33 → −0.30** (losers own MORE) | **Clean composition signal: losers over-buy DECAY jokers (Ice Cream/Gros Michel/Popcorn) that fade. A concrete bot weakness.** |
| **sum_levels / max_hand_level: +0.21/+0.16 → +0.38/+0.36** | Winners LEVEL MORE (broadly + higher peak). NOT concentration — `n_types_leveled` is HIGHER for winners (refutes the "level one hand" part of B). |
| **compounder: +0.15 → +0.20** (small, grows late) | Thesis A is REAL but SECONDARY — winners own only ~0.2 more compounders. Not the dominant lever. |
| build-capacity growth/ante | **Winners sustain ~1.55×; losers collapse 1.55×→1.26× late.** The build-power death is a late growth-rate collapse — confirmed on the deployed bot. |

**Verdict: the win/loss lever is overall BUILD POWER driven by a COMBINATION — economy + more
leveling + avoiding decay jokers + modestly more compounders — NOT primarily entering an exotic
scaling basin (thesis A's compounder d is only 0.20).** So the from-scratch core's build-value is
the right framing, but it should value *sustained capacity growth + economy + leveling + decay-
avoidance* (thesis B-weighted), with compounder acquisition as a secondary term. The S0 kill-switch
should test realizable mid-game build *improvement broadly*, not compounder construction specifically.

**Immediate near-term candidate (independent of the big core):** losers over-invest in DECAY jokers
(d −0.49 at ante 4). The current build-value scores them at ~peak, not decayed/expected value. A
decay-aware penalty in `build_scoring.py` is a cheap, low-risk paired A/B — worth trying before the
multi-week core. (Caveat: partly correlational — losers may grab weak jokers out of desperation — so
gate on a real CRN winrate A/B, not the d alone.)

## 4. Architecture (the build-value core)

Reused as bedrock (zero/low effort): exact forward sim + `full_sim_verification_gate.py`; the v3
encoder (`ml/encoding.py`); the greedy play core (`play_scoring.py`, near-optimal per the audit);
`winrate_bench_par.py` / `winrate_bench_config.py` + CRN paired A/B ≥96 seeds.

Four new pieces, in dependency order:

1. **`age_build_to_ante(build, k)` — forward-projection shim.** Deterministic closed-form
   extrapolator: project each owned/candidate scaling joker's counter to ante k at its per-type
   accrual rate; project hand-levels forward at the run's realized planet rate concentrated on the
   committed hand. **This is the one genuinely missing piece** and the thing that gives growth a
   gradient. It gets its own standalone correctness gate (S1).
2. **Trajectory build-VALUE = per-candidate Δ clear-capacity vs the deterministic wall schedule.**
   `Δ = Σ_{k=now..8} clearcap(age_build_to_ante(build+cand, k), wall_k) − clearcap(age_build_to_ante(build, k), wall_k)`.
   The wall schedule is *deterministic data* (RNG-horizon-safe). This is the only framing where buy-A
   and buy-B differ *by construction* (the inertness that killed clearcap-as-leaf was current-counter,
   owned-state, global). Re-cast the clearcap head to consume *projected* inputs.
3. **Shallow commitment search — change the LEAF, not the depth.** Keep the existing depth-2 beam;
   swap in the Δ-trajectory leaf. Deeper search is a *gated micro-experiment only* (A11 showed
   depth-2→3 regressed 5→0; the play audit shows deeper search amplifies leaf error). Optional: score
   2-shop lookahead over *sampled* future shops (expectation over RNG, never a fixed deep tape).
4. **Forced-scaling-basin data.** Re-capture with **per-round joker counters + hand_levels logged**,
   generated from a forced-exploration policy (ε toward buying+feeding an anchor) so the value finally
   *observes* the strong basin instead of re-learning the weak one (the on-policy trap that closed
   every prior framing). Supervised retrain — no value bootstrapping (FVI collapsed), no whole-run
   backward solver (intractable + RNG-fictional), no tabula-rasa self-play (data-starved on 8 cores).

One play change: point the discard/clear-line targeting + playstyle bonus at the committed anchor's
hand type once an anchor is owned (cheap; does not touch the near-optimal greedy clear logic).

## 5. Staged build plan (each stage has a kill-condition; the program dies cheap if the thesis is wrong)

| Stage | What | Gate (kill condition) | Cost (8 cores) |
|---|---|---|---|
| **S-pre — free split check** | Mine the existing 226 graft rows (`value_buildgate*`) + a fresh deployed-bot capture: characterize outs as additive vs compounder, and winner-vs-loser late growth/hand-concentration. | If outs+winners are additive/concentration-dominated → thesis B (economy/concentration), not A. Pick the core accordingly. | Hours, no new code. |
| **S0 — realizability kill-switch** (§6) | On ~40 deployed loss seeds, build a *new* bounded buy-SEQUENCE search at antes **3–5 shops** (RNG-validated horizon), re-sim each purchase forward, roll to terminal; does a winning line exist and is it *compounder-anchored*? | Winning replay-verified line for **≥40%** of losses AND those lines own ≥1 compounder by ante 5 (not just a bigger additive pile). Else: thesis A false → pivot to B or stop. **Validity gate is PRIMARY: only count lines whose decisive buy lands at ante ≤5 (validated tape); ante 6–8 tail = expectation-with-variance, flagged low-confidence.** | **~1 week build** (new sequence-search + sampler-rollout harness — NOT a cheap reuse of `endgame_out_test.py`) + overnight run. |
| **S1 — projection-shim correctness + DISCRIMINATION** | Build `age_build_to_ante`; validate vs realized trajectories AND across-candidate spread. | (a) projected counter/level at ante k within ~25% median rel-err of realized values in the 214 caps reaching ante 7; **(b) the projection moves buy-A vs buy-B vs end-shop by a margin exceeding the win-value 0.0 inertness baseline on the same cached visits.** (b) makes S2 a genuinely independent test. | 1 day re-capture gen + hours. |
| **S2 — non-flatness hard pre-gate** | Dump per-candidate Δ across ≥200 real shop visits. | Std across candidates within a visit materially > 0 (vs the win-value 0.0 / clearcap-leaf ~0 baseline) AND the anchor tops the Δ in >55% of S0-constructable states. **If flat, the build-value path is dead — off-ramp.** | Hours, no training. |
| **S3 — leaf-swap A/B (shallow)** | Graft the Δ-trajectory leaf behind a flag; CRN paired A/B vs deployed. | Winrate > 22–26% at ≥96 seeds, paired-delta LCB > 0, AND late growth shifts toward the winner signature. A11 micro-gate: depth-2 ≥ depth-1 on this leaf. | Hours–1 day/A·B. |
| **S4 — forced-scaling flywheel** (only if S3 wins) | Regenerate from the winning policy; retrain; re-A/B; close distribution shift. | Retrained value holds AUC ≥ deep-ante baseline on the NEW distribution; second A/B beats S3. | **1–2 days × N rounds = 1–2 weeks** (the only genuinely expensive stage, gated behind a proven S3 gain). |

## 6. The first decisive experiment (corrected)

**S0 — can a bounded mid-game (ante 3–5) buy-SEQUENCE search realizably construct a winning,
compounder-anchored build on loss seeds the live bot fails?** This is the one load-bearing test, and
it is *not* the broken out-test (which grafts one additive joker at the terminal blind with zero
runway). It fixes that instrument's four defects: fork at antes 3–5 (runway + validated RNG), search
a sequence (tests the trajectory claim), roll forward through the exact sim to real run-completion,
and characterize the winning lines (additive vs compounder → distinguishes thesis A from B).

**Honest cost correction:** this is a *new* sequence-search + sampler-rollout harness (~1 week
build), not an overnight reuse. Its decisive region (antes 6–8) overlaps the unvalidated RNG tail, so
the ante≤5 validity gate is a *primary* requirement, not a footnote: a winning line only counts if
its decisive construction lands on the validated tape. Pre-test for free first (S-pre) against the
226 existing graft rows.

## 7. Honest outlook

- **Realistic landing:** ~**30–40%** white if S0 passes and S2 clears non-flatness — a real break
  past 26%, driven by entering basins the bot structurally refuses.
- **Superhuman (~80%) is NOT reachable** here: (a) the seed-faithful RNG is validated only to ~ante
  5, but the deaths are at antes 6–8 where the model is untrusted; (b) 8 physical cores / ~400
  runs/hr forecloses AlphaZero-scale self-play; (c) even a strong (sub-oracle) endgame player
  recovers only ~10% — the residual gap is build construction under partial information, the hardest
  remaining problem.
- **Probability it also caps low: ~45–55%.** Case for caps-low: 81% additive outs, play near-dead,
  archetype flat, and the load-bearing projection shim + deep-RNG region are unbuilt/unvalidated
  exactly where they must work. Case for break-through: all four documented failures share ONE root
  cause (current-counter / next-boss / global-state / single-graft myopia), and this is the first
  design that attacks all four at once *and* gates itself to die cheap (days→weeks, not a quarter).
  No prior probe ever tested mid-game realizable *construction*.
- **Off-ramps:** S0 finds additive-only → pivot to **economy-curve + one-hand-concentration** (thesis
  B; smaller ~28–30% ceiling but data-supported). S2 flat → stop the build-value path at near-zero
  cost; residual lever is META (skips/tags/vouchers, never systematically optimized). Projection
  shim fantasy → bound the value to ≤ante 5 + treat 6–8 as expectation-with-variance.

**Recommendation:** run S-pre (free, hours) then decide A vs B, then S0 (~1 week) as the cheap
go/no-go before any net/flywheel — the same die-cheap discipline that made S0-foresight worth 1.5 hr
instead of a quarter.
