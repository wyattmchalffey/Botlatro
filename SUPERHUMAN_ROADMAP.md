# SUPERHUMAN_ROADMAP.md

*2026-06-09. Synthesis of a 17-agent audit (7 repo auditors + external research + adversarially-verified
diagnosis) over every plan, post-mortem, kill-gate, and benchmark in this repo. Every load-bearing number
below was re-verified against the artifact that produced it.*

## Executive summary

The bot is at **19.5% held-out sim winrate** (39/200, confirmed twice). The defensible superhuman bar on
White Stake is **≥95% over ≥1,000 random seeds** — the developer's stated position is that perfect play
essentially always wins White Stake, experienced players self-report ≥95%, and no proven-unwinnable
white-stake seed has ever been exhibited. The gap is ~75pp and it is almost entirely **policy**, not variance.

Three months of rigorous kill-gates proved something real but narrower than the project concluded: the bot
is at a verified **local optimum of a myopic architecture** — every *per-decision* surface (shop picks,
mid-game build selection, reroll/economy, decay, static skips) is near-optimal *under a value function whose
horizon is one blind*. That is not an architecture ceiling for the codebase; it is the ceiling of per-decision
argmax. The assets needed to break it — a 99.9%-exact sim, ~98% seed-faithful RNG prediction, a Rust core,
10⁴–10⁵ runs/day of data-gen — exist and have never been composed into (1) a whole-run planner or (2) a
learning loop that can exceed its teacher. Those two compositions are the path. Everything else is ±2-8pp.

**One urgent correction first: every reported winrate and every kill-gate is contaminated by a foresight
leak.** The deployed play heuristic reads `state.known_deck[:draw_count]` — the exact ordered future draws —
at 12+ sites (`bots/basic_strategy/draw_evaluation.py:63,198,253,562,588,608,743`, `discard_state.py:32`,
`score_projection.py:55`). In sim, `known_deck` is the true draw order (`local_runner.py:2337`); on the
bridge it is empty (`api/state.py:478`). So 19.5% is the winrate of a *clairvoyant-draws* player, no
no-foresight ablation exists anywhere in the repo, and the near-optimality kill-gates (the 14.9% play
oracle, the 73.7% out-test, the construction kill-switch, the flat value-leaf) all certified a clairvoyant
player. Fixing the map is Phase 0.

---

## Part 1 — The bar (external evidence, 2026-06-09)

| Anchor | Stake | Number |
|---|---|---|
| Developer (localthunk, via Steam discussion) | White | perfect play wins essentially every time |
| Experienced-player self-reports (Steam threads) | White | ≥95%; "90% skill / 10% luck" |
| Steam global achievements | — | 72.1% of all owners have won ≥1 run; 12.2% ever beat Gold |
| Top players (DrSpectred/Haelian streak content) | Gold | 9–15 win streaks ⇒ inferred ~60-80%/run |
| Slay the Spire genre calibration (TerrenceM / ForgottenArbiter) | A20H | top human 40-72%; optimal-play estimate ~60-73% |
| BalatroBench best LLM (gemini-3-pro) | White, 5 fixed seeds | 9/15 (60%) — n=15, fixed public seeds, not comparable to random-seed winrate |

**Bar: ≥95% white-stake winrate on ≥1k random seeds, information-set-honest (no seed/future reading at
decision time), certified on the faithful sim and spot-checked on the live bridge.** A "clearly beats strong
humans" intermediate bar is ~90%. No audited human per-run dataset exists; if a rigorous claim is ever
needed, the human baseline must be measured (tracked runs or VOD scraping) — the "~80% human" figure in
PROGRESS.md is unsourced and probably too low.

Prior-art notes: nobody else has this repo's sim+RNG moat (the public ecosystem drives the real game via
balatrobot). DemoEvolve (arXiv 2605.24539) got its Balatro gains from a handful of expert demonstration
trajectories stabilizing a self-improvement loop — sparse win/loss feedback alone misled it; that is this
repo's shop-ranker failure in miniature. The proven superhuman formula in stochastic single-player games is
**learned afterstate value + shallow chance-aware search, trained on self-play outcomes at scale** (2048 TD
n-tuple ~72% to 32768; Stochastic MuZero) — exactly the component class this project has never built.

## Part 2 — Corrections to the project's own beliefs

These are places where memory/docs drifted from what the artifacts actually show. Delete the stale versions.

1. **"~42% of late deaths are play-recoverable" is refuted by this repo's own oracle.** The death-margin
   *proxy* (score-ratio threshold) said ~42%; the verified depth-6/width-6 play-oracle fork at the death
   blind (`.data/endgame_play_audit.json`, n=316) clears **14.9%** of losses (10.5% at ante 8). Play is a
   +3-8pp lever, not a co-equal pillar.
2. **The construction kill-switch v1 numbers (1.7% vs 5.6%, "~8% recoverable / 92% unrecoverable") were
   retracted** — bias artifact of a ~61%-clean subset. Definitive v2 (contamination-free): intervention
   5.0%/attempt ≈ neutral reroll 4.8%; **26.5% of losses flippable** by some clean single perturbation.
   Selection near-optimality stands; the magnitude story changed.
3. **"Shop is closed" is proven only for per-decision shop micro-optimization** under the myopic value
   function. The 73.7% out-test shows winning purchases existed in the offer stream (caveat: 9/12 late
   losses were slot-full, so realizable headroom is lower but includes sell+buy routes). Coordinated
   multi-shop *sequence* planning (NEW_CORE_PLAN §6) was specified and never executed. Don't reopen shop
   micro-ranking; shop decisions *under a whole-run plan* are a different, untested surface.
4. **S0 "foresight unlearnable" closed only the reactive route.** Proven: early observable state cannot
   predict the winning basin (AUC 0.41-0.51 = chance) because the deciding information (ante 4-8 offers)
   is not in the state. Not tested: a planner that *reads* future offers via the now-98%-faithful seed
   predictors (offline/labels), or a policy that *shapes* the basin instead of predicting it (self-play —
   suit concentration had zero variance under the baseline because the baseline never stacks suits; the
   null is conditional on that policy class).
5. **"~22-26% is the architecture ceiling"** is the ceiling of *the current heuristic per-decision bot*,
   measured with a 4-archetype hand taxonomy, a sub-oracle play search (oracle median ratio 0.556 < bot's
   own 0.718), and single-perturbation tests. It says nothing about what the sim + seed predictors + Rust
   core + planner composition can reach.
6. **Reported winrates are clairvoyant-draws upper bounds** (foresight leak, see exec summary) and benches
   default to the generic sampler sim (`faithful=False`), with no p-values/CIs anywhere, ~11pp MDE at the
   habitual 100-seed size, and several knobs rejected on noise-level reads (Blueprint fix, dig,
   planet-scaling, safe-margin). The largest bench ever run is 384 runs; a 2pp-powered paired certification
   (~1-2k seeds, one overnight) has never been run. Last live-bridge bench: 2026-05-24, pre-current-config.
7. **Antes 1-2 were never kill-gate tested** (`s0_midgame_construction.py` forks antes 3-5 only), yet the
   only two adopted winrate levers in project history were early-game corrections (+4.5pp safety_base
   antes 1-2; +2-4pp ECON_W from ante-3 money divergence) and the bot is documented over-optimistic at
   antes 1-4 (pred 0.23-0.48 vs actual 0.15-0.17). A +3-8pp early-game residual is plausible and unlisted.
8. **Consumable/leveling allocation is under-audited**: greedy use-immediately with flat biases
   (`held_consumables.py`), while `sum_levels` is a top win/loss discriminator (d +0.38; winners ~level 6
   vs losers ~4). Only 24-seed ablations were ever run on it.

## Part 3 — The ceiling, causally ranked

| # | Cause | Est. locked | Confidence |
|---|---|---|---|
| 1 | **No whole-run planning** — per-decision argmax, one-blind play horizon, one-shop/next-blind shop horizon (ante-8 target blending only kicks in at ante 7, `shop_forecast.py`). Losses accumulate across the run: winners sustain 1.55×/ante build growth vs losers' 1.26×; miss ratio degrades 0.795(a5)→0.651(a8); the info needed to win exists in the run's future and the architecture cannot see it. | ~40-60pp | high |
| 2 | **Determinization unused** — seed-exact RNG (~98% faithful) is execution-only; no search has ever seen future shops/draws/packs. The "deterministic single-agent puzzle" reframe (PHASE8_NEURAL_PLAN) exists on paper only. | 10-30pp (overlaps #1) | medium |
| 3 | **No learning loop that can exceed its teacher** — 100% of neural work was imitation/distillation of the heuristic, mostly aimed at shop (the surface with provably no per-decision headroom), at 384 runs while capacity is 10⁴-10⁵/day. Value net run-starved (train AUC 0.999 / val 0.708). Self-play / search-improved targets (PHASE8 Stages 3-6): never started. | 20-40pp (vehicle for #1) | high |
| 4 | **Deployed bot has no play search at all** — BasicStrategy 1-ply; the d3w2 beam lives only in data-gen. Verified play-oracle headroom 14.9% of losses (lower bound, oracle is sub-bot at ratios). Five bosses (Violet Vessel, Eye, Plant, Manacle, Amber Acorn; 59/316 deaths) are 0% clearable at the death blind but exactly known at every prior shop — boss-prep is unharvested. | +3-8pp | medium |
| 5 | **META surfaces are constants** — never-skip (0% use of 24 fully-modeled tags), pack-open by hardcoded constants, greedy consumables, voucher denylist permanently killing Telescope→Observatory. Only static-rule versions tested (and they lose because skip value depends on the specific forgone shop — the counterfactual version was named and never built). | +2-6pp | low |
| 6 | **Measurement distortion** — foresight leak in every gate; generic-mode benches; no statistical inference; unmeasured human bar; no live-bridge number for the current config. | ±2-4pp + map corruption | high |
| 7 | **Late-ante RNG trust gaps** — divergence audit skipped BUY (2,186 transitions), OPEN_PACK/CHOOSE_PACK_CARD (1,061), SELECT/SKIP_BLIND, stochastic-joker plays (1,046), stochastic bosses (73); bridge-validated only to ~ante 5. Gating risk for any deep-search program (would poison labels exactly where 78-86% of deaths occur). Bounded; Immolate/Ouija exist for cross-checking. | 0pp direct; gates #1-3 | high |

## Part 4 — The roadmap

Sequenced so every phase has a pre-registered kill-gate and de-risks the next. Costs assume this 8-core box
plus optional cheap rented compute (repo's own costing: 10-50k solver-quality runs ≈ $3-50).

### Phase 0 — Fix the instruments (~1 week, blocking everything)

- **P0.1 Kill the foresight leak.** Add a no-foresight mode: `known_deck` reads in `basic_strategy`
  (draw_evaluation, discard_state, score_projection) replaced with the existing draw-odds DP /
  multiset-sampling machinery (hand_viability already has the DP). Re-run the 200-seed bench both ways.
  *This re-baselines the entire project.* Gate: measure the honest number; if honest ≪ 19.5%, blind play
  quality reopens as a lever (it was never actually measured). Also re-run the 14.9% play oracle and the
  out-test roll-forwards honest-mode before trusting them again.
  **✅ DONE 2026-06-10 (`BALATRO_NO_FORESIGHT=shuffle|hide`, bots/no_foresight.py). 1000-seed held-out
  paired certification: clairvoyant 17.8% vs honest 12.4%; d = -5.4pp (CI -8.2..-2.6), McNemar
  p = 0.0002. Dev-seed optimism adds another ~3.7pp (21.5% dev vs 17.8% held-out). The certified
  honest held-out baseline is 12.4% — the true starting point is ~83pp from the bar, and honest play
  quality REOPENS as a lever (the gate fired).**
- **P0.2 Statistical floor.** McNemar p + Wilson CI in `bot_paired_ab.py` / `winrate_bench_config.py`
  (~20 lines). One overnight 1-2k-seed paired-CRN certification of current config vs pre-ECON_W baseline.
  Batch re-test the four noise-rejected knobs (Blueprint, dig, planet-scaling, safe-margin) properly powered.
- **P0.3 Faithful-mode benches by default** (divergence is 2% now; generic mode is no longer earning its
  distribution mismatch).
- **P0.4 Close late-ante RNG. ✅ CERTIFIED 2026-06-12** (live-bridge per-class replay, 5 rounds, 20
  bridge-verified sim bugs). Per-class audit (`scripts/p04_transition_class_audit.py`) over BUY/OPEN_PACK/
  CHOOSE_PACK_CARD/SELECT_BLIND/stochastic-joker/showdown-boss: audit v5 shows **0 score/hand divergences
  across 8 worklist seeds to antes 5-7**, first-ever ante-6+ lockstep in every class. Surviving diffs are
  money-only (dollar-ticker comparator artifact + one gated economy residual on seed 0000039's Fish
  cash-out). VERDICT: certified for Phase B full-trajectory value targets; **gate per-seed on
  `_rng_diverged`**. See [[sim-correctness-baseline]] memory + `.data/p04_rootcause_round{1-5}.md`.
- **P0.5 Pre-register the target.** Superhuman = ≥95% honest white-stake over ≥1k seeds. Decide the
  information regime now: **recommended — information-set-honest deployed bot** (what a human can see;
  no seed reading, no true-future access at decision time); the clairvoyant planner is an *offline*
  diagnostic/label tool only. (A seed-reading live bot would trivialize the claim — tool-assisted, not
  superhuman play.)

### Phase 1 — Whole-run planning (the big lever, ~3-6 weeks)

- **P1.1 Clairvoyant route-oracle (offline diagnostic).** Full-route search with the true future (shops,
  packs, draws) on ~100 honest-mode lost seeds, including sell+buy and skip routes. This measures the
  *realizable routing ceiling* (the 73.7% out-test counted offers, not routes; 9/12 late losses were
  slot-full) and produces hindsight labels: the first decision where the oracle's route diverges.
  **Gate: if clairvoyant routing flips <30% of losses, the whole-run-planning thesis is in trouble — stop
  and re-diagnose. Expected: 50-80% flip.**
- **P1.2 Counterfactual skip decider** (the mechanism PROGRESS.md itself named): at each Small/Big blind,
  sample the forgone shop K times (faithful sampler), compare EV(shop + blind money) vs EV(tag), decide.
  First honest win for META. Gate: paired ≥+2pp, properly powered.
- **P1.3 Multi-shop sequence search** (NEW_CORE_PLAN §6, never executed): plan buys/sells/rerolls across
  this shop + the next 1-2 shops jointly under money/interest/slot constraints, over K determinized
  futures (honest: sampled, not true-seed). The planner's leaf = current heuristic value initially.
  Gate: paired ≥+3pp or kill.
- **P1.4 Boss-prep mode.** Boss identity is visible all ante. When the ante's boss is one of the five
  0%-clearable killers, shop/consumable search optimizes P(clear boss) specifically (Director's Cut reroll
  is currently Violet-Vessel-only, `actions.py:64-66`). Gate: death-share of those five bosses drops.
- **P1.5 Deploy play search late-ante.** Wire the existing d3w2 beam (Rust path) into the deployed bot for
  antes ≥5 with a death-aware objective (maximize P(clear), not EV) at must-clear blinds. The verified
  +3-8pp. Gate: paired ≥+2pp.
  **↑ RESIZED 2026-06-11 (honest replication): honest play-recoverable = 29.7% of losses (vs 14.9%
  clairvoyant), and it extends early (ante-1 losses 85.7% clearable, ante-3 50%). Potential ≈ +13pp,
  ~2x the original budget, and "late-ante only" is the wrong scoping — pair it with honest
  draw-evaluation in the heuristic (replace single-belief-sample peeks with the draw-odds DP).
  These two are now the top winrate levers, ahead of META.**
- **P1.6 Requirement-curve tracking.** The blind schedule is deterministic. Maintain
  projected-build-growth vs required-growth (winners 1.55×/ante, losers 1.26×) as a global feature that
  modulates shop aggression/economy (the existing `_estimated_shop_planning_required_score` only blends
  the ante-8 target from ante 7 — start it at ante 1).

Expected exit: honest winrate in the **30-45%** range. P1 also produces the planner that Phase 2 trains.

### Phase 2 — The learning engine (self-play value iteration, ~2-3 months)

The only component class that exceeds its teacher, and the only one never attempted. The formula that is
unanimous in comparable games: **learned whole-run (after)state value + shallow chance-aware search,
iterated on self-play outcomes.**

- Train V(state) → P(win) (+ auxiliary ante-reached / discounted margin) on **10k-100k on-policy runs of
  the P1 planner bot** (local: ~10k/day; rented: $3-50 per batch). Value-training recipe (CORRECTED
  2026-06-12 — an earlier version of this doc misquoted the record): MC-outcome targets first;
  TD(λ) as a gated experiment (its one win was on-distribution only; its deployment test failed);
  target network was NEVER implemented — it is the textbook stabilizer to TRY against the documented
  fitted-iteration collapse, not a known-working fix. Use the candidate-conditioned architecture
  (the strongest offline head: candidate-subset 12.5× over random — though its one deployment test
  failed as a beam pruner) rather than pooled-state regression. See PHASE_B_ARCHITECTURE.md (v2,
  adversarially reviewed) for the current full design.
- Use V as the planner's leaf (replacing the myopic heuristic leaf at 1-2-ante horizon), then close the
  loop: generate → train → redeploy → regenerate. 2-4 iterations.
- **Why this dodges the Phase 8 graveyard:** (a) outcome labels at 100-1000× the old data scale (384 →
  50k+ runs) attack exactly the run-starvation the curves showed (train 0.999/val 0.708); (b) the net
  never has to out-rank the heuristic per-decision (the failure mode of every shop attempt) — it has to
  value plan-horizon leaves the heuristic cannot see at all; (c) inside search, mis-calibration is far
  less damaging than for argmax overrides; (d) on-policy iteration removes the distribution shift that
  killed the relabel work.
- Gate per iteration: powered paired holdout winrate vs previous iteration. Kill if two consecutive
  iterations are flat.
- Shop decisions get re-attacked here *legitimately* — as plan-horizon value, not micro-ranking.

Expected exit: this is the phase that determines whether the end state is ~50% or ≥90%. No honest way to
predict it from here; the gates will say.

### Phase 3 — Certification (~1-2 weeks)

- 1k+ seed honest certification on faithful sim; live-bridge run of the final config (the sim→real gap has
  never been measured for any current-generation bot).
- Optional: submit to BalatroBench (its 15-run fixed-seed protocol is trivial at this scale) and publish
  the methodology — the sim/RNG moat plus a measured ≥90% honest winrate is a publishable result.

## Part 5 — What NOT to do (all measured, all closed)

- No per-decision heuristic knob tuning (shop/economy/flush/decay/reroll) — genuinely exhausted.
- No shop micro-ranking / shop-leaf value / shop overrides in the current architecture.
- No reactive basin/archetype classifiers from early state (S0: the information is not in the state).
- No static skip/META rules (A/B'd: -10/100; the value is counterfactual-dependent).
- No imitation-only training on heuristic trajectories (provably capped at the teacher; play policy
  rank-corr -0.68 with the very heuristic it imitated).
- No micro-memoization perf work (measured neutral); the remaining hot cost is irreducible without
  behavior-changing A/Bs.

## Appendix — Key artifacts

- `.data/endgame_play_audit.json` — play oracle: 14.9%/10.5%, ratios 0.556/0.718, n=316
- `.data/endgame_out_test.json` — 73.7% had-out rate, n=57 ante-8 losses, mean 5 outs
- `.data/s0_oracle_white_200.json` — archetype oracle +6.5%, flush-dominated
- `scripts/s0_midgame_construction.py` + PROGRESS.md 2026-06-09 — v2 kill-switch 5.0% vs 4.8%, 26.5% flippable
- `scripts/phase8_death_margin.py` — the (superseded-as-evidence) 42% proxy
- Foresight leak: `bots/basic_strategy/draw_evaluation.py:63` etc.; `sim/local_runner.py:2337`; `api/state.py:478`
- Bench power: 100-seed unpaired MDE ~11pp; paired 2pp ≈ 1-2k seeds ≈ 2.5-5h overnight; never run
- Data-gen: SolverPolicy ~9-10k runs/day local; basic bot ~21k/day; sim.step = 1.0% of bot CPU
- NEW_CORE_PLAN.md §6 — the multi-shop sequence search spec (never executed)
- PHASE8_NEURAL_PLAN.md Stages 3-6 — the self-play loop spec (never started)
