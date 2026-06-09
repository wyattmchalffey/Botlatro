# Phase 8–15: The Superhuman Bot (Neural-Guided Search)

**Status:** Active (started 2026-05-31). This is the end-goal architecture
plan — the path from "exact forward model + hand-tuned heuristic search" to a
**superhuman, self-improving, neural-guided planner.** It supersedes the
*offline-solver-as-end-goal* framing in
[`PHASE7_OFFLINE_SOLVER_PLAN.md`](PHASE7_OFFLINE_SOLVER_PLAN.md): the solver
and `basic_strategy_bot` are now **bootstraps**, not the destination.

**Cross-refs:** [`PLAN.md`](PLAN.md) (phases 8–15), the Rust core
([`RUST_PORT_PLAN.md`](RUST_PORT_PLAN.md)), the sim/RNG foundation
([`PHASE7_OFFLINE_SOLVER_PLAN.md`](PHASE7_OFFLINE_SOLVER_PLAN.md)).

---

## 1. The goal and the key reframe

Goal: a bot that plays Balatro at superhuman strength (win rates and scores
beyond strong human play across random seeds and high stakes).

**The reframe that makes this tractable:** with seed-faithful RNG, Balatro is a
**deterministic, perfect-information, single-agent planning problem** — once the
seed is fixed, every shop, boss, pack, and draw is known. There is no opponent
and there are no chance nodes. So this is *not* chess (adversarial → minimax);
it is closer to **optimal solitaire / a deterministic puzzle solver**. The right
algorithms are **best-first / A* / IDA* / PUCT with a strong learned heuristic**,
not minimax.

The single hardest part of a strong game engine — a perfect, fast forward model
with make/unmake — is **already built** (`forward_sim`, 99.9% exact, Rust-backed).
What's missing is the *brain*: a learned evaluation and a search that uses it.

**Why "faster" and "stronger" are the same change.** Today the leaf evaluator is
a greedy rollout (`state_value.clear_probability`) plus a saturating hand-tuned
shop score. Rollouts are the expensive *and* weak way to evaluate a position
(28%+ of runtime; can't see late-game scaling → the ante-8 ceiling). Every
strong modern engine deleted rollouts in favor of a **cheap learned static
evaluation** (Stockfish-NNUE; AlphaZero value+policy net). That one change makes
search simultaneously *cheaper* (µs eval vs ms rollout) and *better-guided*
(learned eval captures synergy/scaling). We adopt the same move.

---

## 2. What we already have (the moat)

- **Exact forward model:** `search/forward_sim.py` + `rust/botlatro-core`
  (PyO3), 99.9% exact vs the real game on the audited surface; ~150 jokers, all
  bosses, vouchers, tags, packs modeled against extracted Lua source.
- **Seed-faithful RNG:** `rng/` — deck, shop, boss, vouchers, tags, packs,
  rerolls, mid-hand procs, created-card identities. Makes the game
  perfect-information from a seed string (bridge-validated to ante ~5).
- **Pure-sim runner + dataset CLI:** `sim/local_runner.py`,
  `dataset/cli.py` (multiprocessing, resumable JSONL).
- **Two bootstrap teachers:** `basic_strategy_bot` (~17–23% white win, fast)
  and the beam `SolverPolicy` (~8%, slower). Either can seed imitation data.
- **Methodology that works:** process_time A/B (`scripts/datagen_speed.py`),
  ≥96-seed quality gates, parity tests, pure-Python source-of-truth with
  native accelerators.

The cost prerequisite that usually kills these projects (a trustworthy
simulator) is paid. We build the brain on top.

---

## 3. Target architecture

Five components, mapped to a modern engine:

| Component | Engine analog | Status |
|---|---|---|
| **State encoder** — `GameState` → learnable tensors (raw joker/card/shop identities, editions, counters, hand levels, deck comp), versioned | NNUE/AZ input features | **✓ done (Stage 0)** |
| **Value head** — state → P(win) / expected ante; cheap forward pass replaces the rollout leaf | NNUE static eval | not started |
| **Policy head** — state → action priors; prunes search to top-k | AZ policy prior | not started |
| **Selective search** — best-first / A* / PUCT over the *single-agent* tree, value+policy guided, tree-reuse across in-run decisions | AZ MCTS / αβ search | not started |
| **Self-play loop** — net guides search → search yields better-than-net targets → retrain → repeat | AZ training loop | not started |
| **Substrate** — batched net inference across parallel workers; encoder/eval in/near Rust | Leela batched NN eval | not started |

Design rules (non-negotiable, "done right"):
- **Raw inputs, not heuristic aggregates.** The 2026-05-29 value-head probe
  failed because it re-weighted the heuristic's own collapsed signals. The net
  must see joker identities, editions, counters, deck composition, hand levels,
  shop contents — things the heuristic throws away.
- **Pure-Python source of truth; native/torch are accelerators** behind parity
  tests. The encoder is dependency-free (stdlib) so it tests under `unittest`.
- **Everything versioned** (`ENCODING_VERSION`, model checkpoints carry the
  encoder version) so schema drift is detectable.
- **Keep all seeds** (wins *and* losses) — losses train the value head.

---

## 4. Staged build plan

Each stage ends with a measured gate. Do not advance until the gate is met.

### Stage 0 — Foundations for learning  ✓ COMPLETE (2026-05-31)
- **0.1 State encoder** (`ml/encoding.py`, `ENCODING_VERSION=1`) ✓ — structured,
  versioned, UNK-safe encoding with full joker/card/shop identity + editions +
  counters + hand levels + deck composition. 13 tests green.
- **0.2 Data pipeline** (`ml/dataset.py`) ✓ — `capture_run` stores a thin but
  replay-complete action log + outcome; `replay_states` re-simulates it to
  reconstruct per-step states; `examples_from_capture` encodes + labels them.
  `verify_capture_roundtrip` gate passes (re-sim reproduces the run exactly and
  survives JSON persistence). 8 tests green.
- **0.3 Training harness** (`ml/model.py` + `ml/train.py`, torch) ✓ — set-encoder
  `ValueNet` over the encoding + `collate_states`; BCE win-prob trainer with eval
  split and versioned checkpoints. Overfit gate passes (tiny synthetic set →
  loss 0.000, 100% acc; checkpoint round-trips). 6 tests green. torch 2.12 (cpu)
  installed (already declared in the `ml` extra).

### Stage 1 — Full-sim verification gate
- **1.1** Run the no-live-bridge verification gate before trusting neural labels:
  forward-sim tests, replay/score tests, RNG fixture validators, and score-edge
  fixtures. The gate writes a JSON report that dataset jobs can cite.
- **1.2** Close or explicitly quarantine any sim/RNG gaps that affect shop/build
  training labels.
- **1.3** *Gate:* `scripts/full_sim_verification_gate.py --jobs N` passes and
  records the report under `.data/`.

### Stage 2 — Shop candidate ranker (first neural policy target)
- **2.1** Generate shop candidate records: encoded state + legal candidate actions
  + common-random-number continuation labels.
- **2.2** Train `score(state, candidate_action)`, not `V(state)`. The model learns
  pairwise/listwise action ranking so it is not capped to a single teacher move.
- **2.3** Deploy as a selector over candidate actions generated by the exact
  simulator/search. *Gate:* beats `solver_shop_basic_play_bot` on held-out
  contiguous seed ranges at comparable compute.

### Stage 3 — Policy head + selective search
- **3.1** Use the ranker/policy as priors for policy-pruned best-first / A* / PUCT
  over the single-agent tree; reuse the tree across in-run decisions.
- **3.2** *Gate:* beats Stage 2 winrate at equal or lower compute.

### Stage 4 — Self-improvement loop (the engine of pro-level)
- **4.1** net guides search → record (state, search-visit policy, outcome value)
  → retrain → repeat.
- **4.2** *Gate:* winrate climbs through the 40–50% white-stake mark and keeps
  rising across loop iterations (not just one-shot imitation).

### Stage 5 — Scale & harden
- **5.1** Batched net inference across the worker fleet; encoder/eval hot path
  in Rust if profiling demands. **5.2** Bigger nets, more self-play iterations,
  more seeds. **5.3** Continuous eval harness with fixed seed pools + regression
  gates. *Gate:* stable >50% white, reproducible.

### Stage 6 — High-stakes & pro-human push
- **6.1** Generalize past white stake. **6.2** Stake-specific fine-tuning,
  challenge modes. **6.3** Benchmark vs strong human play / known seed solutions.
  *Gate:* high-stake win rates exceeding strong human expectations.

---

## 5. Risks & mitigations
- **Cross-platform determinism** (for cloud data-gen): integer-exact RNG + IEEE
  floats → likely fine, but gate every new environment with the test suite + a
  same-seed trajectory diff.
- **Encoder omissions** silently cap the model. *Mitigation:* 0.1 gate requires
  field coverage; bump `ENCODING_VERSION` on any change.
- **Net inference cost** could eat the rollout savings. *Mitigation:* keep the
  net small; batch across workers; measure per-decision, not per-run.
- **Self-play instability / reward hacking.** *Mitigation:* outcome-grounded
  value target (won/ante), fixed-seed eval gate every iteration, keep the
  heuristic as a safety baseline.

## 6. Methodology (carried from the project's hard-won lessons)
process_time A/B is ground truth; ≥96 seeds for quality; cProfile lies (use
`scripts/phase_timing.py`); parity-test every native/learned replacement against
the pure-Python path before trusting it.

## 7. Current step & log
- **2026-05-31:** Plan created. Started **Step 0.1 (state encoder)** —
  `src/balatro_ai/ml/encoding.py` + `tests/test_ml_encoding.py`.
- **2026-05-31:** **Stage 0 COMPLETE** — 0.1 encoder (`ml/encoding.py`), 0.2 data
  pipeline (`ml/dataset.py`), 0.3 value net + trainer (`ml/model.py`,
  `ml/train.py`). 27 ml tests green; overfit gate fits to loss 0.000 / 100% acc;
  torch 2.12.0+cpu installed. The later 2026-06-04 audit superseded the
  value-leaf-first target with the sim-gated candidate-ranker path below.
- **2026-06-04:** Reframed after failed value-head/shop-leaf experiments and
  held-out bot checks. Raw `V(state)` is no longer the next target; shop/build
  learning starts with candidate ranking. Added `scripts/full_sim_verification_gate.py`
  and the first candidate-ranker data path (`ml/shop_candidate_dataset.py`,
  `scripts/phase8_shop_candidate_dataset.py`). The full no-live-bridge gate passed
  and a tiny 2-record, 2-worker candidate JSONL smoke completed.
- **2026-06-05:** Candidate-ranker probe found a label-collapse bug before scale:
  same-horizon rollout survivors all returned flat ante values, so the argmax was
  mostly deterministic candidate order. Fixed `rollout_value_after_action` to use
  ante/win survival as the primary value plus a bounded shop/build quality bonus
  for same-horizon survivors. Added duplicate-state filtering to the data CLI and
  a soft rollout-value loss to the ranker trainer. Added an env-gated
  `RankerGuidedShopBot` wrapper (`BALATRO_SHOP_RANKER_CKPT`) for downstream A/Bs.
  Quick sim gate still passes after the changes. Tiny deduped probe: 8 raw captured
  states -> 4 unique rows, mean best margin 0.0866, best actions include pack-0,
  pack-1, and buy-card-1. One-seed wrapper smoke loaded the checkpoint but regressed
  badly, as expected for a four-row model; the checkpoint is a pipeline artifact only.
- **2026-06-05 continued:** Data generation now supports ante-range capture filters,
  deterministic shuffled selection from deduped states, parallel capture jobs, and label
  diagnostics for source/ante coverage, split-half best-action stability, nonzero margins,
  and top-tie counts. Ante-2 parallel smoke (`rollouts=2`, `max_actions=6`) captured 32
  states -> 30 unique, selected 4, mean margin 0.1292, nonzero-margin rate 0.75, but
  split-half agreement only 0.5 and mean top-tie count 2.25. Scale gate: real training
  data should use at least 4 CRN rollouts, track regret/ties, and avoid judging by raw
  top-1 alone.
- **2026-06-05 late:** User correctly pointed out that early shops often have multiple
  winning build basins, so there may be no single "right" answer. Added acceptable-action
  metrics (`mean_actions_within_0_05/0_10`, heuristic-within-band rates, ranker near-best
  accuracy). The 8-row, 4-rollout ante-2 probe confirms this: average 2.25 actions are
  within 0.05 of best and 3.0 within 0.10. Hard-label training overfits top-1 and has worse
  held-out regret; soft mean-pool training is the best tiny variant by held-out regret.
  Do not deploy these checkpoints; use them to justify larger soft/regret-labeled data.
- **2026-06-05 Rust speed checkpoint:** fixed the env-gated Rust best-play path rather than
  working around it. The bridge now mirrors Python Blueprint/Brainstorm and copied
  Swashbuckler semantics; Rust scoring now matches Python for shape-gated jokers, debuffed
  face checks, debuffed scored-card joker triggers, and Eye/Mouth boss zeroing. With the
  conservative joker bailout list removed, normal full-vector parity over four real
  trajectories reports 6,150 best-play calls, 92.6% Rust fast-path use, and 0 divergences.
  Single-seed `basic_strategy_bot` timing improved from 25.5s to 8.27s with
  `BALATRO_RUST_BESTPLAY=1` (~3.1x). Treat this as a trustworthy data-generation speed lane,
  still protected by `scripts/bestplay_parity_check.py` before broad sweeps.
- **2026-06-05 Rust-backed candidate data checkpoint:** re-ran the full sim gate after the
  Rust fixes (`.data/full_sim_gate_after_rust_bestplay_fix.json`, passed). The candidate
  dataset CLI now enables `BALATRO_RUST_BESTPLAY=1` before worker imports and records
  `rust_bestplay` in metrics; `--no-rust-bestplay` is an opt-out for debugging. Added
  source-balanced state selection so small multi-teacher datasets do not accidentally become
  one-teacher data. Balanced ante-2 smoke:
  `.data/phase8_shop_candidates_ante2_r4_8_after_rust_balanced.jsonl`, 8 rows, 4/4 source
  split, 228 candidate continuations in 99.64s, split-half best agreement 0.625. Tiny soft
  ranker train is a pipeline artifact only, not deployable.
- **2026-06-05 overnight stopping point:** generated the first 32-row balanced Rust-backed
  artifact: `.data/phase8_shop_candidates_ante2_r4_32_after_rust_balanced.jsonl`
  (`rollouts=4`, `max_antes=1`, 12 workers, 16/16 source split). It produced 980 candidate
  continuations in 324.22s (3.02/s), with split-half best agreement 0.5625 and average
  2.78 actions within 0.05 of best. Soft-vs-hard ranker sanity check on a 26/6 seed split:
  hard labels memorized train and failed val (`near_best_0_05=0.0`, regret 0.3101), while
  soft labels were weak but better (`near_best_0_05=0.3333`, regret 0.2052). Next morning:
  scale records first; do not deploy this checkpoint.
- **2026-06-05 resume-safe scale prep:** candidate-data generation now writes ordered partial
  JSONL and partial metrics during the expensive label pass (`--partial-every`, default 4).
  Added `--resume-partial`, keyed by `(source_bot, seed, state_index)`, so interrupted
  multiworker runs can skip already-labeled shop states on restart. Focused tests pass
  (`18 passed`). A 4-row multiprocessing smoke wrote final and partial artifacts; the
  resume smoke reused all 4 records (`resumed_partial_records=4`, `remaining_label_jobs=0`).
  Next scale command should include `--resume-partial` from the start.
- **2026-06-05 128-row scale gate:** generated
  `.data/phase8_shop_candidates_ante2_r4_128_after_rust_balanced.jsonl` with 128 balanced
  ante-2 rows, 4 CRN rollouts, 12 workers, Rust best-play on, and resume checkpoints.
  The run finished cleanly: 1,019 captured states -> 964 deduped -> 128 selected, 3,828
  candidate continuations in 948.22s (4.04/s), 64/64 source split, split-half best
  agreement 0.6562, and average ambiguity of 2.875 actions within 0.05 of best. Added
  heuristic baselines to ranker metrics and validation-regret checkpoint selection.
  Best selected mean model: val regret 0.1062, near-best@0.05 0.5833. Best selected
  attention model: val regret 0.1042, near-best@0.05 0.5833. Same-split heuristic baseline:
  val regret 0.0859, near-best@0.05 0.6957. Verdict: no neural deployment yet; the model is
  fitting train but not beating the held-out heuristic. Next gate is larger/cleaner labels,
  not another wrapper A/B.
- **2026-06-05 label-quality + horizon-2 branch:** added label-quality metadata to ranker
  examples (best margin, split-half agreement, near-best action counts), training-split
  quality filters, and repeated split-sweep tooling
  (`scripts/phase8_ranker_split_sweep.py`). Filtering the 128-row horizon-1 labels did not
  help; margin and stable/low-ambiguity filters became data-starved and still lost to the
  heuristic on held-out regret. A small horizon-2 label probe
  (`.data/phase8_shop_candidates_ante2_r4_m2_32_after_rust_balanced.jsonl`) was slower
  (916 continuations in 596.14s, 1.54/s) and flatter (`mean_best_margin=0.0190`,
  4.06 actions within 0.05), but much more split-half stable (`0.9375`). Attention trained
  on this 32-row horizon-2 artifact beat the heuristic across repeated splits:
  mean val regret 0.0503 vs 0.1066, regret wins 7/7, near-best@0.05 0.7517 vs 0.6000
  (`.data/phase8_ranker_sweep_ante2_r4_m2_32_attention.metrics.json`). The same sweep on
  the 128-row horizon-1 artifact lost on average: 0.1079 vs heuristic 0.0969. Verdict:
  horizon depth, not label filtering, is the next promising lever. Scale horizon-2 before
  any neural shop deployment.
- **2026-06-05 64-row horizon-2 confirmation:** generated
  `.data/phase8_shop_candidates_ante2_r4_m2_64_after_rust_balanced.jsonl` with 64 balanced
  ante-2 rows, 4 CRN rollouts, `max_antes=2`, 12 workers, Rust best-play, and resume
  checkpoints. It finished cleanly: 634 captured -> 601 deduped -> 64 selected, 1,928
  continuations in 1,237.53s (1.56/s), 32/32 source split. Labels remained flat but
  stable-ish: `mean_best_margin=0.0297`, 3.89 actions within 0.05, split-half agreement
  0.8281. Attention repeated split sweep
  (`.data/phase8_ranker_sweep_ante2_r4_m2_64_attention.metrics.json`) reports mean model
  regret 0.0729 vs heuristic 0.1217, regret wins 7/7, near-best@0.05 0.6576 vs 0.5847
  (wins 4/7). This confirms horizon-2 attention is the first neural shop-ranking branch
  that beats the heuristic offline beyond a tiny single split. Still do not deploy yet;
  next gate is 128-row horizon-2 or a mixed ante-2/3 horizon-2 dataset, followed by the
  same split sweep and then downstream ranker-guided shop A/B if it survives.
- **2026-06-05 combined horizon-2 + online smoke:** added repeated `--data` support and
  deduped multi-file loading so existing horizon-2 artifacts can combine into a 96-row gate.
  Added candidate-action filtering so training can match deployment action space, plus
  wrapper env gates for action types and max neural actions per shop. Combined 96-row
  horizon-2 attention is strong offline: unfiltered action-space sweep regret 0.0535 vs
  heuristic 0.0960 (wins 7/7), safe-action sweep (`buy,open_pack,end_shop`) regret 0.0463
  vs heuristic 0.0756 (wins 7/7), near-best@0.05 0.7899 vs 0.6524. But the online wrapper
  remains fragile: unconstrained live ranker was catastrophic due to SELL/REROLL chains;
  safe action gating fixes that failure mode, yet deterministic 24-seed paired A/B is only
  slightly positive (ranker 6 wins vs baseline 5, mean ante +0.083, better 9 / worse 10 /
  same 5; `.data/bot_paired_solver_shop_ranker_h2_96_ante2_safeactions_pyhash0_24.json`).
  Verdict: no promotion. Continue scaling horizon-2 labels or add ante-3 coverage, then run
  48+ deterministic safe-action A/B only after repeated split metrics remain ahead.
- **2026-06-05 overnight ante-balance checkpoint:** added `--balance-antes` to candidate
  state selection and verified the focused suite (`78 passed`). The attempted mixed ante-2/3
  horizon-2 artifact
  `.data/phase8_shop_candidates_ante2to3_r4_m2_64_after_rust_balanced.jsonl` completed
  cleanly: 64 rows, 1,864 candidate continuations in 1,293.63s, 32/32 source split, no
  stderr. It did not add ante-3 data: `records_by_ante={"2": 64}`. Likely cause is the
  collector filling each seed's `--per-seed 2` quota with ante-2 shops before ante-3 shops
  are collected. Keep it only as extra ante-2 horizon-2 data. Next morning fix collection
  coverage first, either with ante-balanced collection, higher `--per-seed`, or explicit
  `--min-capture-ante 3 --max-capture-ante 3`, before training a mixed-ante ranker.
- **2026-06-05 mixed-ante v2 + online gate:** fixed collection-side ante balancing and added
  label value version 2, which preserves same-horizon economy/build resource value so richer
  safe lines can beat slightly higher immediate headroom when both survive. Added
  dataset-time `--candidate-action-types` filtering, which made the deployable safe-action
  artifact cheaper and cleaner:
  `.data/phase8_shop_candidates_ante2to3_r4_m2_64_resourcefloor_safeactions.jsonl` has
  64 rows, exact 32/32 source and ante split, all label v2, and 1,348 continuations in
  1,227.81s. Mean-encoder safe-action sweep is promising offline: regret 0.0599 vs heuristic
  0.1168, regret wins 5/7, near-best@0.05 0.8132 vs 0.7560. But online replacement still
  loses: best 12-seed deterministic smoke with cap=1 and margin=1.0 is 3/12 wins vs baseline
  4/12, mean ante -0.167. Also fixed a wrapper bug where `BOOSTER_OPENED` reset the per-shop
  ranker action cap. Added a baseline-comparison gate that probes the wrapped solver on a
  deep copy and only permits neural safe-action overrides that score above the baseline
  candidate, but it still loses on the same 12-seed lane: 2/12 wins vs baseline 4/12,
  mean ante -0.25
  (`.data/bot_paired_solver_shop_ranker_ante2to3_safeactions_resourcefloor_mean_cap1_margin10_comparebaseline_12.json`).
  Verdict: do not promote ranker replacement. Next target is an override/advantage model
  that labels the baseline action/continuation directly and only overrides when predicted
  candidate-minus-baseline advantage is large.
- **2026-06-05 advantage-objective gate:** added `baseline_index`, `baseline_value`, and
  candidate-minus-baseline advantages to shop-ranker examples/batches, plus `--loss
  advantage_mse` and thresholded advantage override metrics. The objective is better aligned
  offline: mean-encoder repeated split sweep on the 64-row mixed-ante safe-action artifact has
  mean lift `+0.0906` over the baseline action, regret delta `-0.0906`, and positive lift in
  6/7 splits
  (`.data/phase8_ranker_sweep_ante2to3_r4_m2_64_resourcefloor_safeactions_mean_advantage_mse.metrics.json`).
  But the trained checkpoint still fails online at a usable threshold: margin `0.10` gives
  2/12 wins vs baseline 4/12 and mean ante -0.417
  (`.data/bot_paired_solver_shop_ranker_ante2to3_safeactions_resourcefloor_mean_advantage_mse_baseline_margin010_12.json`).
  Margin `0.30` is exactly neutral, indicating safety only when neural overrides are almost
  entirely suppressed. Trace on regression seed `0300005` shows an ante-2 Jumbo Celestial
  override over `end_shop` with predicted margin `0.27193`, which derails the future
  shop/economy line. Verdict: the advantage objective is the right deployment framing, but
  the current horizon-2 labels are still too shallow. Next data gate: deeper/full-run
  baseline-vs-candidate advantage labels, especially for pack-open and end-shop tradeoffs.
- **2026-06-05 deep advantage pipeline smoke:** fixed the baseline-coverage mismatch by adding
  dataset `--include-heuristic-action`, train/sweep `--keep-heuristic-action`, and live wrapper
  comparison-only baseline scoring. The wrapper can now score the solver's filtered-out action
  (for example `reroll`) while only returning safe neural overrides from
  `BALATRO_SHOP_RANKER_ACTION_TYPES`. Focused tests pass (`90 passed`). A 4-row `max_antes=8`
  smoke with safe candidates plus retained solver action wrote
  `.data/phase8_shop_candidates_deep_advantage_includeheuristic_smoke.jsonl`: 50 candidate
  continuations in 370.57s, exact source/ante balance, `heuristic_present_rate=1.0`, one retained
  `reroll` baseline outside the safe action set, and mean best margin 1.0851. Tiny train smoke
  with `--loss advantage_mse --keep-heuristic-action` verifies end-to-end metrics at 100%
  baseline coverage. Cost is high, so scale this lane carefully: 16 rows first with resume
  checkpoints before any larger run.
- **2026-06-05 focused deep-label budgeting:** added `--candidate-priority deep_advantage`,
  which puts `end_shop`, pack opens, and buys ahead of low-priority actions before `--max-actions`
  truncation. A comparable 4-row focused smoke with `--max-actions 4` wrote
  `.data/phase8_shop_candidates_deep_advantage_focused_smoke.jsonl`: 38 candidate continuations
  in 283.05s, same selected-state balance and `heuristic_present_rate=1.0`, one retained
  comparison-only `reroll` baseline, and about 24% less wall time than the 50-continuation smoke.
  Focused tests pass (`91 passed`). Next scale lane should use this focused budget:
  `--candidate-action-types buy,open_pack,end_shop --candidate-priority deep_advantage
  --max-actions 4 --include-heuristic-action --max-antes 8 --max-steps 1200 --rollouts 2`.
- **2026-06-05 8-row focused deep artifact:** added metrics for retained heuristic action type
  distribution and outside-safe comparison baselines. Generated
  `.data/phase8_shop_candidates_deep_advantage_focused_8.jsonl`: 8 rows, 70 candidate
  continuations in 327.31s (`0.214/s`) with 8 workers, exact source/ante balance,
  `heuristic_present_rate=1.0`, heuristic action types `buy=7, reroll=1`, and outside-safe
  baseline rate `0.125`. Combining this with the 4-row focused smoke for a tiny repeated
  split check (`.data/phase8_ranker_sweep_deep_advantage_focused_12_mean_advantage_mse.metrics.json`)
  is encouraging but not deployable: model regret wins 7/7, mean advantage lift `+0.8644`,
  but harmful override rate `0.1429`. Next gate: generate at least a 16-row focused deep
  artifact with the same flags and judge by repeated-split advantage harm/lift before online A/B.
- **2026-06-05 16/28 focused deep artifact and v3 label fix:** generated
  `.data/phase8_shop_candidates_deep_advantage_focused_16.jsonl`: 16 rows, 150 candidate
  continuations in 929.50s with 8 workers, exact source/ante balance, and `heuristic_present_rate=1.0`.
  It includes comparison-only solver baselines outside the safe neural set (`sell=5`, `reroll=2`),
  which is the missing comparison signal. The 28-row repeated split sweep
  `.data/phase8_ranker_sweep_deep_advantage_focused_28_mean_advantage_mse.metrics.json` improves
  average regret on all splits (`0.8469` model vs `1.4799` heuristic; mean advantage lift
  `+0.4281`), but it is still not deployable because harmful overrides average `0.3905`.
  Audited the label semantics for economy: v2 used resource/economy mostly as a floor, so same-depth
  survivors could still collapse toward immediate/build score. `LABEL_VALUE_VERSION=3` now adds an
  explicit bounded resource bonus for same-horizon survivors, preserving the "safe clear plus more
  money is better" signal. Focused tests pass (`92 passed`). Regenerate v3 focused deep labels before
  the next scale run or online A/B.
- **2026-06-05 v3 32-row combined gate:** generated two independent v3 focused deep shards,
  `.data/phase8_shop_candidates_deep_advantage_focused_v3_16.jsonl` and
  `.data/phase8_shop_candidates_deep_advantage_focused_v3_16b.jsonl`, both exact 8/8 source and
  ante splits. Combined quality is still noisy (`split_half_agreement_rate=0.46875`), but the
  attention ranker now clearly beats the heuristic as a candidate scorer:
  `.data/phase8_ranker_sweep_deep_advantage_focused_v3_32_attention_advantage_mse.metrics.json`
  reports regret `0.7233` vs heuristic `0.9500`, near-best@0.05 `0.4741` vs `0.3367`, and top-1
  `0.4095` vs `0.2721`. Override deployment is still unsafe: at threshold `0.1`, mean lift is
  only `+0.1559`, positive in 2/7 splits, with harmful override rate `0.3476`; threshold `0.3`
  reduces harm to `0.2143` but is positive in only 1/7 splits; threshold `0.5` remains harmful
  (`0.2857`). Training only on split-half-stable rows starves the model and is worse. Verdict:
  no online A/B and no checkpoint promotion. Next gate is cleaner labels, not threshold tuning:
  relabel these states or a smaller fresh focused shard with more CRN rollouts, then require lower
  harmful override before live testing.
- **2026-06-05 r4 probe + snapshot relabel efficiency fix:** a fresh 8-row v3 focused
  `rollouts=4` shard
  (`.data/phase8_shop_candidates_deep_advantage_focused_v3_r4_8.jsonl`) did not improve label
  reliability: 144 continuations in 866.75s, split-half agreement `0.125`, attention regret
  `0.8783` vs heuristic `0.5139`, harmful override rate `0.6429`. Do not blindly scale fresh
  r4 collection. Instead, candidate records now carry a reloadable `state_snapshot`, and
  `scripts/phase8_shop_candidate_dataset.py --input-records` can relabel the same selected
  states with different rollout settings while skipping trajectory collection/deduplication.
  Smoke artifacts `.data/phase8_snapshot_relabel_smoke_source.jsonl` and
  `.data/phase8_snapshot_relabel_smoke_relabel.jsonl` prove the path. Focused tests pass
  (`39 passed`). Next label-quality experiments should generate snapshot-bearing states once,
  then compare CRN counts/settings on the same states.
- **2026-06-05 same-state r2/r4 comparison:** generated
  `.data/phase8_same_state_v3_r2_8.jsonl` and relabeled the exact same states as
  `.data/phase8_same_state_v3_r4_8.jsonl` using `--input-records`. Split-half agreement stayed
  `0.375` for both, but r4 changed the best labeled action on 5/8 states and had large
  shared-candidate value movement (mean abs delta `0.9403`, max `2.2970`). Tiny attention sweeps
  favor r4 on this state set (regret `0.2772` vs heuristic `0.7518`, harmful override `0.0`) over
  r2 (regret `0.4676` vs heuristic `1.1365`, harmful override `0.1429`). Verdict: r2 deep labels
  are too unstable for deployment calibration; r4 may be a better target, but it needs a larger
  same-state snapshot gate before any online A/B.
- **2026-06-05 same-state r4/r8 gate and revised label direction:** generated a reusable
  balanced 16-state snapshot pool (`.data/phase8_capture_pool_v3_16.jsonl`) and relabeled it as
  `.data/phase8_capture_pool_v3_r4_16.jsonl` and
  `.data/phase8_capture_pool_v3_r8_16.jsonl`. R4 took 292 candidate continuations in 1383.48s;
  r8 took 584 continuations in 2858.25s (47.6 minutes), both with 8 workers. Doubling rollouts did
  not improve split-half best-action agreement: both r4 and r8 are `0.25`. R8 changed the best
  action on only 3/16 exact states compared with r4, but shared-candidate values still moved a lot
  (mean abs delta `0.4358`, max `1.4950`), and r8 ranker sweeps do not clear deployment gates
  (attention regret `0.5356` vs heuristic `0.4100`; mean regret `0.4123` vs heuristic `0.4100`;
  thresholded overrides remain unsafe/negative on average). Training only on r8 examples with
  `best_margin >= 0.25` also did not fix calibration. The conclusion changed: the issue is not
  only sampling noise; many early/deep shop states have multiple viable futures, so forcing a
  single argmax "best action" creates fake precision. Next neural work should use the 64-state
  capture-only pool (`.data/phase8_capture_pool_v3_64.jsonl`, captured in 172.06s with exact
  32/32 source and ante splits) for uncertainty/tie-aware labels: acceptable action sets,
  pairwise preferences only above clear margins, or confidence-aware advantage targets.
- **2026-06-06 confidence-aware audit and filter gate:** added paired-CRN confidence diagnostics
  for raw shop candidate records and ranker examples. The new audit script writes
  `.data/phase8_capture_pool_v3_r8_16.confidence.json` and
  `.data/phase8_capture_pool_v3_r4_16.confidence.json` in milliseconds. The r8 pool has only
  `12.5%` high-confidence sampled winners vs runner-up and `87.5%` ambiguous winners; r4 is only
  slightly better (`18.75%` / `81.25%`). Best-vs-heuristic confidence is more useful but sparse:
  only `25%` of r8 states have a practical high-confidence override candidate. Training only on
  confidence-supported baseline improvements is not enough on this tiny set: the filtered
  mean/`advantage_tie_mse` sweep
  `.data/phase8_ranker_sweep_capture_pool_v3_r8_16_mean_advantage_tie_m010_conf_baseline_lcb005.metrics.json`
  trained on 2-3 rows per split and still had negative lift (`-0.1878`) with high harmful override
  (`0.5833`) at threshold `0.0`. Verdict: confidence-aware targets are the right direction, but
  the immediate task is data acquisition/selection for more high-confidence baseline-vs-candidate
  examples, not threshold tuning on the 16-row r8 pool.
- **2026-06-06 targeted snapshot selection:** added
  `scripts/phase8_select_shop_state_pool.py` so expensive relabeling can focus on baseline
  competition states from the cheap 64-state pool. The broad 64-state pool has 41/64 states where
  the solver heuristic action is outside the focused `buy/open_pack/end_shop` candidate set. The
  first targeted 16-state selection is exactly source/ante balanced and has 16/16 outside-candidate
  baselines, but is mostly `sell`. A diverse variant
  `.data/phase8_capture_pool_v3_64_targeted_diverse_16.jsonl` balances source while spreading
  heuristic action types (`buy=4`, `end_shop=4`, `open_pack=2`, `reroll=3`, `sell=3`). Cheap
  r2/short 4-state probes show why diversity matters: the sell-heavy probe produced no
  high-confidence overrides, while the diverse probe found a `25%` practical high-confidence
  best-vs-heuristic rate, including an ante-3 `end_shop` baseline where opening a pack had a large
  positive paired lower bound. Next run should label the diverse 16-state subset at short r2/r4,
  then deepen only states with baseline-vs-candidate confidence instead of relabeling all 64.
- **2026-06-06 adaptive deepening check:** added
  `scripts/phase8_select_deepening_states.py` to convert shallow candidate labels into a tiny
  "deepen next" state pool based on paired candidate-minus-heuristic confidence. The selector now
  records rollout count and supports `--min-rollouts`, because the first adaptive deepening test
  found an r2 false positive. The diverse r2/short smoke selected one apparent ante-3
  `end_shop -> open_pack` opportunity with LCB `1.7621`, but deepening that exact state to
  r4/max_antes=8 took 231.47s and changed the evidence to mean advantage `0.7564`, SEM `1.5167`,
  LCB `-0.7602`. The r4 deepened record is rejected by the same selector at
  `--min-rollouts 4`. Treat r2/short labels as a cheap exploration pass only; train/deploy gates
  need same-horizon r4+ confidence or sequential sampling until the paired lower bound remains
  positive.
- **2026-06-06 sequential probe and cost gate:** added
  `scripts/phase8_sequential_baseline_probe.py`, which compares candidates against the heuristic
  baseline with paired CRN samples, stops candidates early on clear positive/negative confidence,
  and records `sequential_*` audit fields. A per-state wall-clock budget now prevents indefinite
  sequential probes. Focused tests pass (`52 passed`). The first deep 2-state probe
  (`min_rollouts=4`, `max_rollouts=8`, `max_antes=8`) exceeded the 15-minute tool timeout and was
  stopped. A shallow 2-state probe
  `.data/phase8_sequential_baseline_probe_diverse2_r2to4_m4.jsonl` completed in 130.77s with
  2 workers, but only produced 2-rollout timeout records and no high-confidence overrides. Next
  priority is reducing/profiling continuation cost before scaling deep confidence labels.
- **2026-06-06 cost profile and focused confirmation:** added
  `scripts/phase8_rollout_cost_profile.py`. A one-continuation profile showed the solver rollout
  teacher is the cost center: `solver_shop_basic_play_bot` took `18.42s`, with `99.1%` inside
  `choose_action`, while `basic_strategy_bot` took `1.16s` on the same state/action/seed and
  produced the same terminal value. This suggests a two-stage label loop: fast basic-rollout
  exploration, adaptive filtering, then focused solver confirmation. A 4-state basic exploration
  run found high-confidence best-vs-heuristic candidates on 3/4 states in 192.92s; filtering at
  `min_rollouts=4` selected one ante-2 `end_shop -> open_pack` candidate. Added
  `--focus-deepening-candidate` to `phase8_sequential_baseline_probe.py` so solver confirmation
  samples only the chosen candidate plus the heuristic. The focused solver r4/max_antes=8
  confirmation kept the candidate positive with LCB `+0.320` using 8 continuations in 248.02s
  (`.data/phase8_solver_confirm_basic_explore_candidate1_focused_r4_m8.jsonl`). This is the first
  efficient, confidence-gated path that produces a solver-confirmed positive shop override label.
- **2026-06-06 diverse-16 mini-funnel:** ran the two-stage lane over all 16 diverse targeted
  states. Fast basic-rollout exploration completed in 391.82s with 8 workers and identified
  high-confidence practical best-vs-heuristic candidates on 8/16 states
  (`.data/phase8_sequential_baseline_probe_diverse16_basic_r4to8_m8.jsonl`). The adaptive
  selector with `--min-rollouts 4` narrowed this to two candidates, then focused solver
  confirmation completed in 206.73s with 2 workers
  (`.data/phase8_solver_confirm_basic_explore_diverse16_top2_focused_r4_m8.jsonl`). One candidate
  survived as a solver-confirmed positive r4/max_antes=8 label: ante-2 `buy -> open_pack`, mean
  advantage `+1.425`, LCB `+0.645`, positive sample rate `0.75`
  (`.data/phase8_solver_confirmed_positive_labels_diverse16_top2_minr4.jsonl`). The other
  candidate, ante-3 `buy -> end_shop`, was mean-positive but ambiguous. This is still far too
  sparse to train a strong model, but the data-generation recipe is now working and avoids
  putting r2/short false positives into the neural target.
- **2026-06-06 second diverse-16 funnel:** added `--exclude-records` to the targeted selector and
  built a non-overlapping diverse 16-state pool from the remaining 48 snapshots. Basic exploration
  took 390.93s with 8 workers and selected 4 r4-supported confirmation candidates. Focused solver
  confirmation took 258.77s with 4 workers and produced one additional confirmed positive label:
  ante-2 `end_shop -> buy`, mean advantage `+2.151`, LCB `+1.006`, positive sample rate `0.75`
  (`.data/phase8_solver_confirmed_positive_labels_diverse16b_top4_minr4.jsonl`). Two candidates
  were solver-rejected and one stayed ambiguous. Combined funnel rate is now 32 explored states ->
  6 solver-confirmed candidates -> 2 confirmed positives. Sparse, but much cleaner than the old
  one-best datasets.
- **2026-06-06 third diverse-16 funnel:** selected
  `.data/phase8_capture_pool_v3_64_targeted_diverse_16c.jsonl`, excluding both prior diverse
  pools. Basic exploration took 399.16s with 8 workers and produced 230 candidate continuations
  across 15 labeled records; one selected state (`0410021`, state 38) did not produce a probe
  record, so future selection should avoid states that cannot yield at least two executable probe
  actions. The adaptive `min_rollouts=4` filter selected 3 solver-confirmation candidates. Focused
  solver confirmation took 262.94s with 3 workers and produced one additional confirmed positive
  label: ante-2 `end_shop -> buy`, mean advantage `+0.912`, LCB `+0.224`, positive sample rate
  `0.75` (`.data/phase8_solver_confirmed_positive_labels_diverse16c_top3_minr4.jsonl`). The
  ante-3 `buy -> open_pack` candidate was solver-rejected, and ante-3 `sell -> buy` remained
  mean-positive but ambiguous. Combined funnel rate is now 47 fast-explored records -> 9
  solver-confirmed candidates -> 3 confirmed positives. This confirms the label path works but is
  sparse enough that the next data iteration should produce acceptable sets/pairwise margins, not
  another top-1-only ranker target.
- **2026-06-06 confidence-aware advantage target:** extended shop-ranker examples and batches with
  per-candidate paired confidence fields against the heuristic baseline, then added
  `confidence_advantage_tie_mse`. Clear positive/negative confidence intervals keep their signed
  candidate-minus-baseline target; ambiguous intervals train as zero/ties. The train and split
  sweep scripts accept the new loss and report positive/negative/ambiguous confidence-label
  counts. The 9 solver-confirmed records from the three diverse funnels are exactly balanced at
  margin `0.10`: 3 positive, 3 negative, 3 ambiguous labels. A tiny split sweep on those records
  (`.data/phase8_ranker_sweep_solver_confirm_9_confidence_advantage_tie_mse.metrics.json`) is
  intentionally not promoted: attention regret is `0.829` vs heuristic `0.543`, mean lift is
  `-0.286` at threshold `0.0`, and harmful override rate remains `0.381`. The useful outcome is
  that the safer target is now wired and test-covered (`112 passed`); the next blocker is more
  solver-confirmed comparison data, not loss plumbing.
- **2026-06-06 fourth diverse-16 funnel and 15-label sweep:** selected the final non-overlapping
  16-state slice from `.data/phase8_capture_pool_v3_64.jsonl`
  (`.data/phase8_capture_pool_v3_64_targeted_diverse_16d.jsonl`). Because it is the remainder,
  heuristic action diversity is limited (`buy=12`, `sell=4`), but ante coverage stays balanced.
  Fast basic exploration completed 16/16 labels in 398.06s with 8 workers and selected 6
  r4-supported candidates. Focused solver confirmation completed in 291.76s with 6 workers and
  produced one additional confirmed positive: ante-2 `buy -> end_shop`, mean advantage `+0.743`,
  LCB `+0.198`, positive rate `0.75`
  (`.data/phase8_solver_confirmed_positive_labels_diverse16d_top6_minr4.jsonl`). The four-pool
  label pool is now 63 fast-explored records -> 15 solver-confirmed candidates -> 4 confirmed
  positives, with a confidence-label mix of 4 positive, 4 negative, and 7 ambiguous
  candidate-vs-baseline examples. The 15-example `confidence_advantage_tie_mse` sweep
  (`.data/phase8_ranker_sweep_solver_confirm_15_confidence_advantage_tie_mse.metrics.json`) is
  better than the 9-example smoke on near-best/top-1, but still not deployable: mean regret is
  `0.578` vs heuristic `0.480`, attention regret is `0.638` vs heuristic `0.480`, and mean lift
  remains negative. The next scaling move should generate a larger capture pool beyond these 64
  states, then reuse the same two-stage funnel and confidence target.
- **2026-06-06 fresh 128-state capture pool:** generated
  `.data/phase8_capture_pool_v3_128_fresh.jsonl` from seed offset `420000`, using 8 workers and
  capture-only selection. It produced 128 exact-balanced source/ante states in 187.26s from 1,024
  captured / 1,001 deduped states. The solver heuristic distribution is broad (`buy=52`,
  `sell=35`, `end_shop=23`, `reroll=7`, `open_pack=6`, `use_consumable=5`) and 70/128 heuristic
  actions sit outside the focused safe-action candidate set. During first-slice selection, fixed
  `phase8_select_shop_state_pool.py` so `balance_fields` balances marginal field counts rather
  than only full tuple groups; the first fresh targeted slice is now exact 8/8 source, exact 8/8
  ante, and spans all six heuristic action types
  (`.data/phase8_capture_pool_v3_128_fresh_targeted_diverse_16a.jsonl`). Fast exploration of that
  slice took 393.99s with 8 workers and selected one r4-supported candidate. Solver confirmation
  turned it into an ambiguity, not a positive: ante-2 `use_consumable -> end_shop`, mean `+0.025`,
  LCB `-0.823`. The combined confidence pool is now 16 solver-confirmed comparisons: 4 positive,
  4 negative, 8 ambiguous. The 16-example sweep
  (`.data/phase8_ranker_sweep_solver_confirm_16_confidence_advantage_tie_mse.metrics.json`) is
  still below promotion gates, though mean-encoder near-best/top-1 beat the heuristic. Continue
  slicing the fresh 128 pool; do not promote until held-out lift turns positive and harmful
  overrides fall sharply.
- **2026-06-06 fresh slice B:** selected a second non-overlapping 16-state slice from the fresh
  128 pool with exact source/ante balance and all six heuristic action types represented
  (`.data/phase8_capture_pool_v3_128_fresh_targeted_diverse_16b.jsonl`). The cheap basic pass was
  strong, with practical high-confidence best-vs-heuristic signals on 11/16 states, and the
  adaptive filter selected 8 solver-confirmation candidates. Strong solver confirmation produced
  no positives: 2 negative-UCB rejects, 5 ambiguous/max-rollouts, and 1 state timeout at three
  paired samples. The combined confidence-label pool is now 24 records: 4 positive, 6 negative,
  14 ambiguous. The 24-example confidence-aware sweep
  (`.data/phase8_ranker_sweep_solver_confirm_24_confidence_advantage_tie_mse.metrics.json`) is
  closer but still below promotion: mean encoder regret `0.328` vs heuristic `0.286`, mean lift
  `-0.042`, and harmful override rate `0.286`. This reinforces the current recipe: use cheap
  exploration only as a proposal mechanism, and keep solver-confirmed negatives/ties in the
  training target so the model learns when not to override.
- **2026-06-06 fresh slice C:** selected a third non-overlapping 16-state slice from the fresh
  128 pool, excluding slices A/B. Source and ante balance remained exact 8/8; heuristic-action
  coverage narrowed with the remaining pool (`buy=4`, `end_shop=5`, `open_pack=1`, `reroll=2`,
  `sell=4`). Fast exploration took 463.52s with 8 workers and selected 3 r4-supported
  confirmation candidates. Solver confirmation took 212.54s with 3 workers and produced one new
  confirmed positive: ante-3 `end_shop -> open_pack`, mean advantage `+1.493`, LCB `+1.124`,
  positive rate `1.0`
  (`.data/phase8_solver_confirmed_positive_labels_fresh128_diverse16c_top3_minr4.jsonl`). The
  combined confidence-label pool is now 27 records: 5 positive, 6 negative, 16 ambiguous. The
  27-example confidence-aware sweep
  (`.data/phase8_ranker_sweep_solver_confirm_27_confidence_advantage_tie_mse.metrics.json`)
  produced the first small positive aggregate validation lift: mean encoder regret `0.302` vs
  heuristic `0.343`, mean lift `+0.041`, and 4/7 positive lift runs; attention lift is only
  `+0.004`. Still do not promote: harmful override rates remain high (`0.298` mean, `0.255`
  attention), and the label pool is still heavily ambiguous.
- **2026-06-06 fresh slice D:** selected a fourth non-overlapping 16-state slice from the fresh
  128 pool. The remaining pool has narrowed to `buy/end_shop/sell` heuristic baselines, but source
  and ante stayed exact 8/8. Fast exploration took 394.31s with 8 workers and selected 2
  r4-supported confirmation candidates. Solver confirmation took 148.53s with 2 workers and
  produced zero new positives; both cheap-positive candidates became ambiguous/max-rollouts. The
  combined confidence-label pool is now 29 records: 5 positive, 6 negative, 18 ambiguous. The
  29-example sweep
  (`.data/phase8_ranker_sweep_solver_confirm_29_confidence_advantage_tie_mse.metrics.json`) keeps
  the mean encoder slightly positive (`+0.018` lift at threshold `0.0`, `+0.027` at threshold
  `0.10`) but still unsafe (`0.335` harmful override rate raw, `0.271` at threshold `0.10`).
  Attention falls negative on aggregate lift. Treat slice D as no-override calibration data, not
  promotion evidence.
- **2026-06-06 fresh slice E and covered-harm metric:** selected a fifth non-overlapping 16-state
  slice from the fresh 128 pool. The selector saw the remaining pool narrowed to `buy=38`,
  `end_shop=8`, `sell=18` heuristic baselines and chose exact 8/8 source/ante coverage with
  `buy=6`, `end_shop=5`, `sell=5`. Fast exploration took 490.21s with 8 workers and selected one
  r4-supported confirmation candidate. Solver confirmation took 147.10s and produced a strong
  positive: ante-3 `buy -> open_pack`, mean advantage `+1.842`, LCB `+1.470`, positive rate `1.0`
  (`.data/phase8_solver_confirmed_positive_labels_fresh128_diverse16e_top1_minr4.jsonl`). The
  combined confidence-label pool is now 30 records: 6 positive, 6 negative, 18 ambiguous. The
  30-example sweep
  (`.data/phase8_ranker_sweep_solver_confirm_30_confidence_advantage_tie_mse.metrics.json`) is
  positive on aggregate lift for both encoders (`+0.042` mean, `+0.048` attention) but still
  unsafe. Added covered-state helpful/harmful override rates to the ranker metrics and sweep
  summaries; the mean threshold sweep
  (`.data/phase8_ranker_sweep_solver_confirm_30_mean_thresholds.metrics.json`) shows why promotion
  is premature: threshold `0.0` has lift `+0.042` but harmful covered rate `0.179`, while threshold
  `0.5` lowers harmful covered rate to `0.029` but loses lift (`-0.013`). Next priority is more
  confirmed comparisons and better calibration, not deployment.
- **2026-06-06 train-calibrated threshold check:** added train-side threshold calibration to
  `phase8_ranker_split_sweep.py`. For each split, the sweep now evaluates candidate thresholds on
  the train side, picks the highest-lift threshold under a harmful-covered-rate cap, then reports
  held-out validation behavior at that chosen threshold. With a `0.05` cap on the 30-record
  mean-encoder sweep
  (`.data/phase8_ranker_sweep_solver_confirm_30_mean_thresholds_calibrated.metrics.json`),
  selected thresholds averaged `0.093`; validation lift was only `+0.007` and harmful covered rate
  was still `0.163`. This says the current score scale is not calibrated enough to choose a safe
  override gate from training labels. Use the ranker as a data-prior/search-prior for now, not as
  a deployed shop override.
- **2026-06-06 fresh2 pool and 34-record sweep:** generated
  `.data/phase8_capture_pool_v3_128_fresh2.jsonl` from seed offset `430000`, selecting 128
  exact-balanced source/ante states from 1,024 captured / 986 deduped states in 270.25s. The first
  fresh2 targeted slice restored broad heuristic-action coverage
  (`buy=3`, `end_shop=3`, `open_pack=3`, `reroll=3`, `sell=3`, `use_consumable=1`) with exact
  source/ante balance. Fast exploration took 385.00s with 8 workers and selected 4 r4-supported
  confirmation candidates. Solver confirmation took 334.01s with 4 workers and produced one new
  confirmed positive, one negative, one ambiguous, and one partial/timed-out candidate. The positive
  was an ante-3 pack-target choice (`open_pack -> open_pack`, likely different pack index), mean
  advantage `+2.223`, LCB `+0.831`, positive rate `0.75`
  (`.data/phase8_solver_confirmed_positive_labels_fresh2_diverse16a_top4_minr4.jsonl`). The
  combined confidence-label pool is now 34 records: 7 positive, 7 negative, 20 ambiguous. The
  34-record sweep
  (`.data/phase8_ranker_sweep_solver_confirm_34_confidence_advantage_tie_mse.metrics.json`) is the
  strongest result so far: mean encoder raw lift `+0.143` with harmful covered rate `0.109`, and
  attention raw lift `+0.132` with harmful covered rate `0.071`. The mean encoder's train-calibrated
  gate under a `0.05` train harmful-covered cap now transfers much better than before: validation
  lift `+0.126`, harmful covered rate `0.048`. Still not deployment-ready because calibrated mean is
  positive in only 3/7 splits and the confirmed label pool is tiny, but this is the first result
  where a safety-gated neural override looks directionally plausible.
- **2026-06-06 fresh2 slice B and 37-record sweep:** selected a second non-overlapping fresh2
  targeted slice
  (`.data/phase8_capture_pool_v3_128_fresh2_targeted_diverse_16b.jsonl`) with exact source/ante
  balance and 10/16 heuristic actions outside the focused candidate family. Fast basic exploration
  took 402.21s with 8 workers and again found cheap practical high-confidence signals on 8/16
  states. The r4 filter selected three ante-2 `end_shop` candidates against buy/open-pack
  baselines, but focused solver confirmation took 232.75s with 3 workers and made all three
  ambiguous (`max_rollouts`, no high-confidence positives). This is useful no-override calibration:
  cheap exploration is still too eager to skip early shop tempo. The combined confidence-label pool
  is now 37 records: 7 positive, 7 negative, 23 ambiguous. The 37-record sweep
  (`.data/phase8_ranker_sweep_solver_confirm_37_confidence_advantage_tie_mse.metrics.json`) keeps
  neural lift positive, with attention now best overall: regret `0.376` vs heuristic `0.523`, raw
  lift `+0.146` positive in 6/7 splits, and threshold `0.25` lift `+0.114` with harmful covered
  rate `0.079`. Train-calibrated attention lift is `+0.118` positive in 5/7 splits, but held-out
  harmful covered rate remains `0.102`, so the model is still a search/data prior, not a deployed
  override.
- **2026-06-06 fresh2 slice C and build-forward selector filter:** added
  `--candidate-action-types` / `--exclude-candidate-action-types` to
  `phase8_select_deepening_states.py` so solver-confirmation budget can target build-forward
  opportunities instead of repeating known-noisy early `end_shop` proposals. Focused selector tests
  pass. Slice C kept exact source/ante balance, but the remaining fresh2 pool has narrowed to
  `buy/end_shop/open_pack/sell` heuristic baselines. Fast exploration labeled 15/16 records in
  414.65s and produced only two practical high-confidence override states. The build-only filter
  selected one ante-3 `end_shop -> buy` candidate; solver confirmation made it mean-positive but
  ambiguous (mean `+1.092`, SEM `1.295`, LCB `-0.204`). The combined confidence-label pool is now
  38 records: 7 positive, 7 negative, 24 ambiguous. The 38-record sweep
  (`.data/phase8_ranker_sweep_solver_confirm_38_confidence_advantage_tie_mse.metrics.json`) keeps
  the neural signal positive: attention regret `0.310` vs heuristic `0.465`, raw lift `+0.156`,
  and train-calibrated lift `+0.124` positive in 5/7 splits. Calibrated harmful covered rate
  improved to `0.081`, but this is still above the intended `0.05` safety cap.
- **2026-06-06 fresh2 slice D and pack-open caution data:** selected the next non-overlapping
  fresh2 slice with exact source/ante balance, but the remaining pool has narrowed to
  `buy/end_shop/sell` baselines. Fast exploration looked promising, with practical
  high-confidence best-vs-heuristic on 7/16 states, and the build-only filter selected three
  `open_pack` candidates against `end_shop`/`sell` baselines. Solver confirmation made all three
  ambiguous: mean advantage `+0.435`, SEM `1.374`, LCB `-0.939`, no high-confidence positives.
  The combined label pool is now 41 records: 7 positive, 7 negative, 27 ambiguous. The 41-record
  sweep stayed positive but weakened: mean encoder raw lift `+0.062`, calibrated lift `+0.033`;
  attention raw lift `+0.051`, calibrated lift `+0.008`. Harm remains above the cap. The lesson is
  sharp: cheap exploration is overconfident on pack-open tempo as well as early `end_shop`, so
  more ambiguous confirmation data alone is not enough. The next useful label acquisition should
  either target states with stronger solver-confirmed positive priors or improve the cheap
  proposal filter before spending more solver rollouts.
- **2026-06-06 cheap-vs-solver audit and SEM gate:** added
  `phase8_deepening_confirmation_audit.py`, plus optional `--max-sem` and
  `--min-lcb-sem-ratio` filters to `phase8_select_deepening_states.py`. The 41-record audit
  (`.data/phase8_deepening_confirmation_audit_41.metrics.json`) confirms the acquisition issue:
  cheap LCB overlaps heavily between solver positives, negatives, and ambiguities, but cheap SEM is
  much lower for confirmed positives on average. Retrospectively, `max_sem=0.45` keeps 5 proposals
  with 4 positives, 0 negatives, and 1 ambiguity; `max_sem=0.55` is already only 4/8 positives, and
  `max_sem=0.80` admits most negatives. On fresh2 A-D build-forward proposals, `max_sem=0.45`
  selects zero candidates, which means it would have avoided the recent wasted solver
  confirmations. Next acquisition should use strict SEM-gated positives first, then intentionally
  schedule separate ambiguity/no-override collection if calibration needs it.
- **2026-06-06 retrospective strict-SEM pass and 44-label merge:** added exclusion-aware
  deepening selection so already-confirmed candidates are skipped when mining old cheap probes.
  Across all existing cheap exploration outputs, `max_sem=0.45` plus build-forward candidate
  filtering found only three still-unconfirmed candidates
  (`.data/phase8_allcheap_unconfirmed_buildonly_sem045_minr4.jsonl`). Focused solver confirmation
  finished those in 149.13s with 3 workers and produced one confirmed positive, two ambiguous
  labels, and zero negatives
  (`.data/phase8_solver_confirm_allcheap_unconfirmed_buildonly_sem045_top3_focused_r4_m8.jsonl`).
  The positive was ante-3 `sell -> open_pack`, mean advantage `+1.145`, LCB `+0.603`, positive
  rate `0.75`. Also fixed ranker JSONL loading to merge multiple focused confirmations of the
  same state instead of deduping away later candidate actions. The merged sweep now has 42 unique
  state examples but 44 candidate labels: 8 positive, 7 negative, 29 ambiguous
  (`.data/phase8_ranker_sweep_solver_confirm_44_merged_confidence_advantage_tie_mse.metrics.json`).
  Attention has the best raw lift so far (`+0.226`, 6/7 positive split runs), and train-calibrated
  lift is also positive (`+0.151`, 6/7), but calibrated harmful covered rate is still `0.149`.
  This is useful signal, not a deployment gate. Next: use SEM-gated acquisition to find more clear
  positives, then revisit calibration once the positive label pool is no longer starved.
- **2026-06-06 two more strict-SEM positives and 46-label sweep:** after excluding the 44-label
  confirmations, the same strict build-forward selector found two remaining low-SEM alternate
  pack choices (`.data/phase8_allcheap_unconfirmed_buildonly_sem045_after44.jsonl`). Focused
  solver confirmation finished in 113.44s with 2 workers and confirmed both as positives:
  `0410006` state 48 `end_shop -> open_pack` at mean advantage `+0.871`, and `0420020` state 50
  `sell -> open_pack` at mean advantage `+1.792`
  (`.data/phase8_solver_confirm_allcheap_unconfirmed_buildonly_sem045_after44_focused_r4_m8.jsonl`).
  The merged ranker sweep now has 46 candidate labels across 42 unique state examples: 10 positive,
  7 negative, 29 ambiguous
  (`.data/phase8_ranker_sweep_solver_confirm_46_merged_confidence_advantage_tie_mse.metrics.json`).
  Mean encoder calibrated lift is the best deployment-style result so far: `+0.199`, positive in
  7/7 split runs, with harmful covered rate `0.115`. That is progress, but it still misses the
  `0.05` safety cap, so the model remains a search/data prior until we either add more clean
  positives or improve calibration.
- **2026-06-06 fresh2 slice E and skip-action false positive:** after the 46-label update, the
  already-paid cheap pool had zero remaining strict build-forward `max_sem=0.45` opportunities.
  Selected fresh2 targeted slice E with exact source/ante balance and heuristic mix `buy=5`,
  `end_shop=6`, `sell=5`
  (`.data/phase8_capture_pool_v3_128_fresh2_targeted_diverse_16e.jsonl`). Fast 8-worker basic
  exploration finished in 385.59s and found 5/16 practical high-confidence states, but zero strict
  build-forward SEM candidates. Allowing `end_shop` produced one strong cheap candidate:
  `0430154` state 50 `buy -> end_shop`, cheap mean `+1.550`, LCB `+1.154`. Solver confirmation
  rejected it: buy mean `7.518`, end_shop mean `5.414`, candidate stopped by `negative_ucb`
  (`.data/phase8_solver_confirm_fresh2_diverse16e_safe_sem045_top1_focused_r4_m8.jsonl`). Adding
  this negative to the 47-label all-action sweep weakened calibration; a build-forward-filtered
  sweep still had positive lift but harmful covered rates above the safety cap
  (`.data/phase8_ranker_sweep_solver_confirm_47_merged_buildforward_confidence_advantage_tie_mse.metrics.json`).
  Lesson: keep buy/open-pack positive acquisition separate from skip/economy actions until the
  `end_shop` proposal filter is much stronger.
- **2026-06-06 fresh2 slice F adds a clean build-forward positive:** selected slice F from the
  remaining fresh2 pool with exact 8/8 source and ante balance
  (`.data/phase8_capture_pool_v3_128_fresh2_targeted_diverse_16f.jsonl`). Cheap exploration used
  8 workers, finished in 385.71s, and produced one strict build-forward SEM survivor:
  `0430019` state 38 `end_shop -> open_pack`, cheap mean `+0.475`, LCB `+0.203` over 6 cheap
  paired rollouts. Focused solver confirmation validated it: mean advantage `+0.860`, LCB
  `+0.458`, positive rate `1.0`
  (`.data/phase8_solver_confirm_fresh2_diverse16f_buildonly_sem045_top1_focused_r4_m8.jsonl`).
  The all-action merged sweep is now 48 candidate labels across 44 state examples: 11 positive,
  8 negative, 29 ambiguous. Attention calibrated lift is `+0.086` with harmful covered `0.127`.
  The build-forward-filtered sweep has 34 labels across 30 state examples and keeps positive raw
  lift, but calibrated harm is still `0.121`
  (`.data/phase8_ranker_sweep_solver_confirm_48_merged_buildforward_confidence_advantage_tie_mse.metrics.json`).
  Next useful move: continue strict build-forward acquisition on slice G/H, or pause acquisition
  briefly to improve score calibration; do not deploy this gate yet.
- **2026-06-06 fresh2 slice G adds caution labels:** selected slice G from the final 32 fresh2
  states (`.data/phase8_capture_pool_v3_128_fresh2_targeted_diverse_16g.jsonl`). The pool is now
  narrow (`buy=13`, `end_shop=3`), though source/ante balance stayed exact 8/8. Cheap exploration
  used 8 workers, finished in 428.08s, and strict build-forward SEM selected two ante-3 buy
  candidates. Solver confirmation rejected one alternate buy as a negative (`0430217` state 38,
  candidate mean `7.386` vs heuristic buy mean `8.457`, advantage `-1.071`, `negative_ucb`) and
  left one `end_shop -> buy` candidate ambiguous despite positive mean (`0430201` state 41,
  advantage `+0.918`, max-rollouts). The all-action merged sweep is now 50 candidate labels across
  46 state examples: 11 positive, 9 negative, 30 ambiguous
  (`.data/phase8_ranker_sweep_solver_confirm_50_merged_confidence_advantage_tie_mse.metrics.json`).
  Calibration improved even without a new positive: mean calibrated lift `+0.150` / harmful
  covered `0.092`, attention calibrated lift `+0.158` / harmful covered `0.102`. Still no
  deployment: safety target is `<=0.05` harmful covered. Label quality is improving because the
  model is getting better no-override/caution examples, not just more positive examples.
- **2026-06-06 fresh2 exhausted; fresh3 started:** selected final fresh2 slice H
  (`.data/phase8_capture_pool_v3_128_fresh2_targeted_diverse_16h.jsonl`). It stayed exact source
  and ante balanced, but all 16 baselines were `buy`. Cheap exploration finished in 416.48s with
  8 workers and produced no strict build-forward `max_sem=0.45` candidates, so no solver
  confirmation was run. Generated fresh3 capture-only pool from seed offset `440000`
  (`.data/phase8_capture_pool_v3_128_fresh3.jsonl`): 128 exact-balanced source/ante states, 1,024
  captured / 994 deduped, 219.12s with 8 collect workers. First fresh3 targeted slice A is ready
  (`.data/phase8_capture_pool_v3_128_fresh3_targeted_diverse_16a.jsonl`) and restores broad
  heuristic-action coverage: `buy=3`, `end_shop=3`, `open_pack=2`, `reroll=3`, `sell=3`,
  `use_consumable=2`. Next run should cheap-explore fresh3 slice A, then use the same strict
  build-forward SEM gate before solver confirmation.
- **2026-06-06 fresh3 slice A confirmation and 52-label sweeps:** cheap exploration on fresh3
  slice A (`.data/phase8_sequential_baseline_probe_fresh3_diverse16a_basic_r4to8_m8.jsonl`)
  completed 14 usable records in 435.50s with 8 workers. Aggregate cheap signal was strong again
  after the narrow fresh2 tail: mean best-vs-heuristic advantage `+1.367`, mean LCB `+0.671`, and
  practical high-confidence rate `0.571`. The strict build-forward `max_sem=0.45` selector kept two
  candidates
  (`.data/phase8_basic_explore_fresh3_diverse16a_deepen_candidates_buildonly_sem045_minr4.jsonl`):
  seed `0440085` state 22 `open_pack -> buy`, and seed `0440204` state 38 `reroll -> open_pack`.
  Solver confirmation
  (`.data/phase8_solver_confirm_fresh3_diverse16a_buildonly_sem045_top2_focused_r4_m8.jsonl`)
  turned them into caution/no-override evidence rather than clean wins: the first was mean-negative
  (`-0.569`) and the second mean-positive (`+0.840`), but both remained confidence-ambiguous. The
  all-action 52-label sweep
  (`.data/phase8_ranker_sweep_solver_confirm_52_merged_confidence_advantage_tie_mse.metrics.json`)
  now has 48 examples and 52 candidate labels: 11 positive, 9 negative, 32 ambiguous. Attention is
  the current best safety/utility tradeoff: calibrated lift `+0.146`, positive in 6/7 splits,
  override rate `0.263`, and calibrated harmful-covered `0.065`. This improves safety versus the
  50-label attention run (`0.102` harmful-covered) but remains above the `0.05` promotion cap. The
  build-forward-filtered 52-label sweep
  (`.data/phase8_ranker_sweep_solver_confirm_52_merged_buildforward_confidence_advantage_tie_mse.metrics.json`)
  improved too, but remains less safe: attention calibrated lift `+0.177`, harmful-covered `0.119`.
  Next acquisition should continue on fresh3 with strict build-forward SEM, while separately
  working on calibration or conservative thresholding before any live shop override.
- **2026-06-06 fresh3 slice B near-miss and 53-label sweep:** selected slice B from fresh3 while
  excluding slice A
  (`.data/phase8_capture_pool_v3_128_fresh3_targeted_diverse_16b.jsonl`): 16 states, exact 8/8
  source and ante balance, heuristic mix `buy=3`, `end_shop=4`, `reroll=4`, `sell=4`,
  `use_consumable=1`. Cheap exploration
  (`.data/phase8_sequential_baseline_probe_fresh3_diverse16b_basic_r4to8_m8.jsonl`) completed
  16/16 records in 417.37s with 8 workers. Aggregate cheap signal stayed strong (mean
  best-vs-heuristic advantage `+1.241`, LCB `+0.620`, practical high-confidence rate `0.563`), but
  strict build-forward selection (`min_rollouts=4`, `max_sem=0.45`) found zero candidates because
  the best-looking candidates mostly timed out at 2-3 paired rollouts. A separate min-3 near-miss
  pass plus focused cheap deepening
  (`.data/phase8_sequential_baseline_probe_fresh3_diverse16b_buildonly_sem045_minr3_focused_basic_r4to8_m8.jsonl`)
  found one relaxed-after-focused solver-confirmation target: seed `0440079`, state 43,
  `end_shop -> open_pack`, cheap mean `+1.252`, LCB `+0.634`, SEM `0.618`. Solver confirmation
  (`.data/phase8_solver_confirm_fresh3_diverse16b_after_focused_lcb050_sem065_top1_focused_r4_m8.jsonl`)
  made it ambiguous/no-override data instead of a positive: mean advantage `+0.061`, SEM `0.790`,
  LCB `-0.729`. The 53-label all-action sweep
  (`.data/phase8_ranker_sweep_solver_confirm_53_merged_confidence_advantage_tie_mse.metrics.json`)
  has 49 examples and 53 candidate labels: 11 positive, 9 negative, 33 ambiguous. Mean encoder is
  now the closest safe-ish gate: calibrated lift `+0.170`, positive in 6/7 splits, override rate
  `0.282`, harmful-covered `0.062`. This still misses the `0.05` promotion cap, but it is the best
  measured harm/lift tradeoff so far. The matching build-forward-filtered sweep remains less safe
  (`.data/phase8_ranker_sweep_solver_confirm_53_merged_buildforward_confidence_advantage_tie_mse.metrics.json`):
  calibrated harmful-covered is about `0.121-0.122`. Treat relaxed-after-focused labels as
  deliberate caution/calibration data, not as a replacement for strict SEM positive acquisition.
- **2026-06-07 fresh3 slice C positive and 54-label sweep:** selected slice C while excluding
  slices A/B (`.data/phase8_capture_pool_v3_128_fresh3_targeted_diverse_16c.jsonl`): 16 states,
  exact 8/8 source and ante balance, heuristic mix `buy=4`, `end_shop=4`, `reroll=4`, `sell=4`.
  Cheap exploration (`.data/phase8_sequential_baseline_probe_fresh3_diverse16c_basic_r4to8_m8.jsonl`)
  completed 16/16 records in 404.98s with 8 workers. Aggregate cheap signal was weaker than slice
  B (mean best-vs-heuristic advantage `+0.884`, LCB `+0.051`, practical high-confidence rate
  `0.313`), and strict `min_rollouts=4`, `max_sem=0.45` selection found zero candidates. A single
  min-3 near-miss strengthened under focused cheap deepening
  (`.data/phase8_sequential_baseline_probe_fresh3_diverse16c_buildonly_sem045_minr3_focused_basic_r4to8_m8.jsonl`):
  `sell -> open_pack`, mean `+0.852`, SEM `0.280`, LCB `+0.572`. Solver confirmation
  (`.data/phase8_solver_confirm_fresh3_diverse16c_buildonly_after_focused_top1_focused_r4_m8.jsonl`)
  kept it positive but near the margin: mean advantage `+0.564`, SEM `0.493`, LCB `+0.071`,
  positive rate `0.5`. The 54-label all-action sweep
  (`.data/phase8_ranker_sweep_solver_confirm_54_merged_confidence_advantage_tie_mse.metrics.json`)
  has 50 examples and 54 labels: 12 positive, 9 negative, 33 ambiguous. It improves utility:
  attention calibrated lift `+0.231` and mean calibrated lift `+0.197`, both positive in 7/7
  splits. Harm is not yet safe: calibrated harmful-covered is `0.070` for both encoders, worse
  than the 53-label mean gate (`0.062`) and still above the `0.05` cap. The 54-label
  build-forward-filtered sweep
  (`.data/phase8_ranker_sweep_solver_confirm_54_merged_buildforward_confidence_advantage_tie_mse.metrics.json`)
  has high lift but high harm (`0.139-0.177`). Continue collecting positives, but do not treat
  build-forward-only filtering as the deployment surface until calibration catches up.
- **2026-06-07 fixed threshold and live wrapper gate:** threshold diagnostics on the 54-label
  all-action/safe-action sweep showed a conservative offline gate: attention at threshold `0.5`
  has lift `+0.149` with harmful-covered `0.026`, and threshold `1.0` has lift `+0.111` with
  harmful-covered `0.008`. Trained a full-data attention checkpoint
  (`.data/phase8_shop_ranker_solver_confirm_54_attention_confidence_advantage_tie_mse_full.pt`),
  then tested it online with baseline comparison, safe action types, ante 2-3, max 4 candidates,
  and one neural action per shop. The 24-seed held-out lane at offset `540000` did not promote:
  margin `0.5` was equal on wins but lower on mean ante (`2/24 -> 2/24`, d_ante `-0.125`), and
  margin `1.0` was worse (`2/24 -> 1/24`, d_ante `-0.542`). Tracing seed `0540006` showed the
  compounding problem: several individually gated ante-2/3 overrides can distort the build line.
  Added `BALATRO_SHOP_RANKER_MAX_ACTIONS_PER_RUN` to cap total ranker overrides per run; focused
  wrapper tests pass (`python -m pytest -q tests\test_search_bot.py -k shop_ranker`, `11 passed`).
  With run cap `1`, margin `0.5` gains one win and loses none (`2/24 -> 3/24`) but still lowers
  mean ante (`6.50 -> 5.96`, d_ante `-0.542`); margin `1.0` stays negative (`2/24 -> 1/24`,
  d_ante `-0.417`). Conclusion: offline confidence gating is promising but not live-safe yet.
  The next label/model work should handle action-distribution mismatch directly, especially buy
  target kinds such as vouchers and repeated pack/economy overrides.
- **2026-06-07 action-kind filter and deployment-distribution lesson:** audited the 54-label
  candidate pool by action kind. The model had only 2 non-heuristic `buy/voucher` labels versus
  14 `buy/card` and 24 `open_pack/pack`, yet live action-type gating let the ranker choose voucher
  buys. Added `BALATRO_SHOP_RANKER_ACTION_KINDS` to the live wrapper and
  `--candidate-action-kinds` to ranker train/sweep loading so offline gates can match the deployed
  target surface. Focused tests pass (`35 passed`). The exact card/pack safe-action sweep
  (`.data/phase8_ranker_sweep_solver_confirm_54_merged_safeactions_cardpack_confidence_advantage_tie_mse.metrics.json`)
  is more conservative than broad safe actions: attention calibrated lift `+0.172` with
  harmful-covered `0.057`; at fixed threshold `0.5`, attention lift `+0.091` with harmful-covered
  `0.038`. Online, card/pack filtering was not enough: first 24-seed block improved slightly
  (`2/24 -> 3/24`, mean ante `6.50 -> 6.58`), but the next 24-seed block regressed hard
  (`6/24 -> 2/24`, mean ante `6.54 -> 5.88`). Combined 48-seed read is negative: wins `8 -> 5`,
  mean ante roughly `6.52 -> 6.23`. Conclusion: the next useful labels should come from the
  deployment distribution itself: run the ranker in compare-baseline/proposal mode, capture the
  exact override candidates it would take on held-out seeds, solver-confirm those disagreements,
  and train against those mistakes before another live promotion attempt.
- **2026-06-07 deployment-disagreement capture smoke:** added
  `scripts/phase8_ranker_override_capture.py` and a sequential-probe fallback for captured focus
  actions, so ranker proposals can be labeled without relying on generic candidate regeneration.
  Focused tests pass (`21 passed, 45 deselected`). On held-out offset `560000`, 8 seeds produced
  4 live-gated disagreements in 53.89s with the full 54-label attention checkpoint
  (`.data/phase8_ranker_override_capture_smoke.jsonl`): 2 `buy/card`, 2 `open_pack/pack`. A full
  horizon confirmation with a 30s state cap was too tight and yielded 0 records; a short 2-ante
  confirmation smoke joined all 4
  (`.data/phase8_ranker_override_capture_smoke_confirmed_h2.jsonl`) and labeled them 0 positive,
  1 negative, 3 ambiguous
  (`.data/phase8_ranker_override_capture_smoke_confirmed_h2_audit.metrics.json`). This is not a
  training set yet, but it proves the deployment-distribution label lane and reinforces that the
  current ranker confidence is not calibrated to real override value.
  Scaling capture to 32 held-out seeds with 8 workers produced a 16-record queue in 75.39s
  (`.data/phase8_ranker_override_capture_560000_32s16.jsonl`): 6 `buy/card`, 10
  `open_pack/pack`, mean baseline margin `0.913`. Short 2-ante confirmation for all 16 completed
  in 72.48s and joined as 2 positive, 1 negative, 13 ambiguous
  (`.data/phase8_ranker_override_capture_560000_32s16_confirmed_h2_audit.metrics.json`). Do not
  train from the h2 triage labels directly; use them to prioritize deeper confirmations on the
  same deployment-disagreement queue.
- **2026-06-07 deployment-disagreement deep labels:** selected the two h2-positive proposals and
  ran focused r4-to-r8, max-ante-8 solver confirmation. Both completed all 8 paired rollouts in
  224.52s with 2 workers and both became ambiguous/no-override labels: 0 positive, 0 negative,
  2 ambiguous, mean solver LCB `-1.599`
  (`.data/phase8_ranker_override_capture_560000_32s16_deepen_h2_pos_confirmed_r4_m8_audit.metrics.json`).
  Adding them to the merged all-action sweep gives 52 examples / 56 labels
  (`.data/phase8_ranker_sweep_solver_confirm_56_merged_deployment_confidence_advantage_tie_mse.metrics.json`):
  12 positive, 9 negative, 35 ambiguous. Calibration did not improve: attention calibrated lift
  `+0.111` with harmful-covered `0.099`; mean calibrated lift `+0.096` with harm `0.108`. The
  matching card/pack-safe sweep
  (`.data/phase8_ranker_sweep_solver_confirm_56_merged_deployment_safeactions_cardpack_confidence_advantage_tie_mse.metrics.json`)
  also remains unsafe under train-calibrated thresholds: attention calibrated lift `+0.129`,
  harmful-covered `0.097`. Fixed attention threshold `1.0` clears harm (`0.018`) but is weak
  (`+0.029`, positive in 3/7 splits). Verdict: this lane is diagnosing the live failure correctly;
  scale deployment-disagreement confirmations rather than promoting another checkpoint.
- **2026-06-07 58-label deployment checkpoint:** deep-confirmed the next two short-horizon
  mean-positive deployment disagreements
  (`.data/phase8_ranker_override_capture_560000_32s16_deepen_h2_meanpos_next4_confirmed_r4_m8.jsonl`);
  both became solver-confirmed positives after 6 paired rollouts, with mean solver LCB `+0.177`.
  The merged all-action 58-label sweep has 14 positive, 9 negative, and 35 ambiguous labels
  (`.data/phase8_ranker_sweep_solver_confirm_58_merged_deployment_confidence_advantage_tie_mse.metrics.json`):
  attention calibrated lift `+0.207`, harmful-covered `0.079`. The exact deployment-safe
  card/pack sweep
  (`.data/phase8_ranker_sweep_solver_confirm_58_merged_deployment_safeactions_cardpack_confidence_advantage_tie_mse.metrics.json`)
  gives the first plausible fixed offline gate: attention threshold `0.5` has lift `+0.133`,
  harmful-covered `0.025`, positive in 6/7 splits. Trained the full-data safe attention checkpoint
  (`.data/phase8_shop_ranker_solver_confirm_58_attention_safe_cardpack_confidence_advantage_tie_mse_full.pt`)
  and ran two fresh 24-seed live A/B blocks with card/pack actions only, ante 2-3, one neural
  action per run, and baseline margin `0.5`. Offset `580000` improved (`4/24 -> 6/24`, mean ante
  `+0.042`), but offset `590000` regressed (`3/24 -> 1/24`, mean ante `-0.208`). Combined
  48-seed read is tied on wins (`7 -> 7`) and worse on mean ante (`-0.083`), so the checkpoint is
  not promotable. Keep it as a proposal/label-acquisition model.
- **2026-06-07 next label source: backward reanalysis from successful trajectories:** capture full
  shop-state snapshots from ante-8 wins and near-wins, branch from the last shop over legal
  alternatives, and roll forward from that snapshot with paired RNG to label late-game choices.
  Then move the branch point earlier shop-by-shop. This should produce much cleaner late-game
  build/economy labels than ante-1 rollouts while still requiring care for survivorship bias.
- **2026-06-07 backward capture implementation:** added
  `scripts/phase8_backward_shop_state_capture.py` to generate capture-only late-shop snapshot rows
  for that reanalysis lane. Focused unit coverage passes
  (`python -m pytest -q tests\test_phase8_backward_shop_state_capture.py`, `2 passed`). A real
  solver capture on 16 fresh seeds at offset `620000` produced 14 ante-8 shop snapshots from 7
  qualifying trajectories in 225.25s with 8 workers
  (`.data/phase8_backward_shops_solver_620000_16_late.jsonl`). A two-record label smoke then
  flowed through `phase8_shop_candidate_dataset.py --input-records` successfully, proving the
  backward snapshots can be branched and rolled forward by the existing multiworker labeler.
- **2026-06-07 backward late-shop label read:** labeled the 14-record `620000` pool with `r=4`,
  one-ante horizon, max 8 actions, and 8 workers
  (`.data/phase8_backward_shops_solver_620000_16_late_r4_h1_m8.jsonl`): mean best-vs-heuristic
  advantage `+0.133`, mean LCB `+0.028`, high-confidence best-beats-heuristic rate `0.429`.
  Captured and labeled a second 8-record winning-run pool at offset `630000`
  (`.data/phase8_backward_shops_solver_630000_32_late_r4_h1_m8.jsonl`), which was more
  split-half stable (`0.75`) but flatter. The combined 22-record backward-only sweep has sparse
  confidence labels (3 positive, 6 negative, 76 ambiguous), yet attention beats the heuristic on
  held-out late-shop regret (`0.126` vs `0.164`) and near-best@0.05 (`0.643` vs `0.381`). Do not
  train a live checkpoint from 22 records; scale this lane, especially near-win terminal states.
- **2026-06-07 near-win targeting:** added `--exclude-wins` to
  `phase8_backward_shop_state_capture.py` and tested it (`4 passed`). A fresh 32-seed capture at
  offset `640000` collected 6 ante-8 non-winning late-shop snapshots and excluded 5 wins from the
  same block. Labeling those 6 with the same `r=4`, h1 setup produced the strongest backward
  signal so far: mean best-vs-heuristic advantage `+0.395`, mean LCB `+0.165`, practical
  high-confidence override-candidate rate `0.333`
  (`.data/phase8_backward_shops_solver_640000_32_late_nearwin_r4_h1_m8.jsonl`). The combined
  28-record backward sweep raises labels to 7 positive, 12 negative, 88 ambiguous; mean encoder
  beats heuristic regret (`0.162` vs `0.207`, 6/7 wins), while attention regresses (`0.213` vs
  `0.207`). Keep scaling near-win capture; do not promote a late-shop checkpoint yet.
- **2026-06-07 backward deepening funnel:** selected 5 high-signal states from the 28-record
  backward pool with candidate-minus-heuristic filters (`mean >= 0.10`, LCB `>= 0.05`, positive
  rate `>= 0.75`, max SEM `0.80`)
  (`.data/phase8_backward_late_28_deepen_select_m010_lcb005_pr075_sem080.jsonl`). Deeper `r=8`,
  h1 confirmation finished all 5 in 745.93s
  (`.data/phase8_backward_late_28_deepen_select_m010_lcb005_pr075_sem080_r8_h1_m8.jsonl`): all 5
  had a non-heuristic best action, 5/5 were practical-positive by mean, and 3/5 were practical
  high-confidence best-vs-heuristic improvements. The exact cheap proposals confirmed as 3
  positive, 0 negative, 2 ambiguous. This validates cheap labels as a selector for expensive
  confirmation, not as direct deployment labels.
- **2026-06-07 second near-win block:** offset `650000` captured 8 ante-8 non-winning late-shop
  snapshots from 4 qualifying runs and excluded 7 wins. Cheap `r=4`, h1 labeling was mostly
  no-override calibration: heuristic within `0.10` on 7/8 states, mean best-vs-heuristic advantage
  `+0.038`, mean LCB `-0.039`, and zero high-confidence override candidates
  (`.data/phase8_backward_shops_solver_650000_32_late_nearwin_r4_h1_m8.jsonl`). The 36-record
  backward sweep now has 7 positive, 17 negative, and 111 ambiguous candidate labels
  (`.data/phase8_ranker_sweep_backward_late_36_r4_h1_m8_confidence_advantage_tie_mse.metrics.json`).
  It is useful calibration but not a checkpoint source: mean encoder lift is tiny and attention
  improves near-best while regressing regret. After excluding the 5 already deepened states, the
  strict selector found 0 remaining unconfirmed opportunities, so keep collecting fresh
  near-win/fringe pools before training another late-shop model.
- **2026-06-07 third near-win block and label overlay fix:** offset `660000` captured 12 ante-8
  non-winning late-shop snapshots from 6 qualifying runs, excluding 3 wins. Cheap `r=4`, h1 labels
  show more branch signal than `650000` but still noisy: heuristic best rate `0.0`, heuristic within
  `0.10` on `0.25`, mean best-vs-heuristic advantage `+0.241`, mean LCB `-0.071`, and only one
  practical high-confidence best-vs-heuristic state
  (`.data/phase8_backward_shops_solver_660000_32_late_nearwin_r4_h1_m8.jsonl`). The expanded
  48-record cheap sweep is not promotable
  (`.data/phase8_ranker_sweep_backward_late_48_r4_h1_m8_confidence_advantage_tie_mse.metrics.json`):
  9 positive, 20 negative, 152 ambiguous labels; mean encoder regresses on regret, attention only
  ties regret while improving near-best. The strict selector found one new unconfirmed opportunity,
  `open_pack` over `end_shop`; focused `r=8` confirmation made it a clean positive with
  best-vs-heuristic advantage `+0.786`, LCB `+0.501`, and exact proposal audit 1 positive / 0
  negative / 0 ambiguous. Fixed `examples_from_jsonl_paths` so deeper duplicate candidate labels
  replace shallow duplicates during dedupe. With the six r8 confirmations overlaid, the 48-state
  sweep improves to mean encoder regret `0.153` vs heuristic `0.157` and attention `0.154` vs
  `0.157`, with near-best `0.438`/`0.446` vs heuristic `0.330`; confidence-gated lift is still too
  weak for a live checkpoint. After excluding all six r8-confirmed states, the strict selector
  finds zero remaining opportunities in this pool.
- **2026-06-07 mixed wins plus near-wins:** changed the backward acquisition stance: keep some
  winning ante-8 trajectories too, because final shops can be the reason a run converts. Offset
  `670000` near-win-only labels selected three cheap `open_pack` opportunities, but r8
  confirmation made all exact proposals ambiguous. Offset `680000` then captured a mixed block
  without `--exclude-wins`: 20 ante-8 snapshots, 12 from wins and 8 from near-wins. Fixed
  `phase8_shop_candidate_dataset.py` so `terminal_won`, `selection_reason`, terminal score/money,
  and shops-from-terminal survive relabeling, and extended `phase8_select_deepening_states.py` to
  balance selected confirmations by terminal outcome. The strict mixed selector found two strong
  candidates, one win and one near-win. R8 confirmation produced one clean positive and one
  ambiguous exact proposal; the positive came from winning seed `0680010`, where `open_pack`
  beat heuristic `sell` by mean `+1.194`, LCB `+1.070`. With all r8 confirmations overlaid, the
  74-state backward sweep now gives mean encoder regret `0.070` vs heuristic `0.118` and
  near-best@0.05 `0.667` vs `0.536`, but gate lift/harm is still not safe enough for deployment.
- **2026-06-07 second mixed block:** offset `690000` captured 24 ante-8 snapshots from 12
  qualifying trajectories, 14 records from wins and 10 from near-wins. Cheap labels were mostly
  calibration/ties, but the terminal-balanced selector found two winning-trajectory `open_pack`
  opportunities. R8 confirmation made both exact proposals positive with mean best-vs-heuristic
  advantage `+0.507`, LCB `+0.255`. With all r8 confirmations overlaid, the 98-state sweep has
  20 positive, 44 negative, and 307 ambiguous labels. The mean encoder still beats heuristic
  regret on average (`0.107` vs `0.131`), but split stability and fixed gates regressed from the
  74-state read. Keep collecting mixed winning/near-winning late shops; do not promote a
  checkpoint from this gate yet.
- **2026-06-07 third mixed block and 122-state overlay:** offset `700000` captured 24 ante-8
  snapshots from 12 qualifying trajectories, 6 records from wins and 18 from near-wins. Cheap
  labels were much hotter than `690000`: heuristic within `0.10` on 12/24 states, mean
  best-vs-heuristic advantage `+0.249`, mean LCB `+0.092`, and practical high-confidence
  best-vs-heuristic rate `0.25`. The strict terminal-balanced selector found 5 fresh opportunities
  after excluding prior confirmations, with 2 from wins and 3 from near-wins. R8 confirmation made
  the exact proposals 4 positive / 0 negative / 1 ambiguous, including all three `open_pack`
  proposals as positives. With all r8 confirmations overlaid, the 122-state sweep has 37 positive,
  51 negative, and 376 ambiguous labels. Both encoders beat heuristic regret on all seven seed
  splits: mean `0.123` vs heuristic `0.156`, attention `0.119` vs heuristic `0.156`. The mean
  encoder has the better safety read (`+0.011` calibrated lift in 6/7 runs; fixed threshold `0.1`
  lift `+0.017` in 7/7 runs), but threshold `0.1` still covers harmful overrides at `0.086`, so
  keep it as a label/proposal model rather than a deployed shop override.
- **2026-06-07 fourth mixed block is a calibration warning:** offset `710000` captured 14 ante-8
  snapshots from 7 qualifying trajectories, 4 records from wins and 10 from near-wins. Cheap
  labels were mostly flat: heuristic within `0.10` on 11/14 states, mean best-vs-heuristic
  advantage `+0.058`, mean LCB `-0.058`, and no practical high-confidence best-vs-heuristic
  cases. The 136-state selector found one marginal near-win `open_pack` over `end_shop`; r8
  confirmation made the exact proposal ambiguous. With 710000 included, the 136-state sweep still
  beats heuristic regret on average, but stability drops to 5/7 splits for both encoders and
  calibration weakens. Treat 122-state as the current best signal read; use 710000 as useful
  calibration/noise evidence and keep future mixed blocks, including wins, but evaluate block
  quality before assuming scale is helping.
- **2026-06-07 block quality gate:** extended `phase8_shop_confidence_audit.py` so cheap label
  blocks get a reproducible `strong_signal` / `weak_or_mixed` / `calibration_only` verdict before
  we spend r8 or training time. The gate keeps terminal win metadata and matched the manual mixed
  block read: `680000` weak/mixed, `690000` weak/mixed, `700000` strong signal, `710000`
  calibration-only. A quality-filtered sweep that kept `700000` plus all r8 confirmations, while
  excluding weak/calibration mixed cheap blocks, produced a denser 83-example pool: 37 positive,
  28 negative, 252 ambiguous labels. It improved average regret lift on a harder filtered set
  (mean `0.135` vs heuristic `0.200`; attention `0.132` vs `0.200`, both 6/7 split wins), and
  attention's calibrated harmful-covered rate fell to `0.027`. It is still too small to deploy,
  but the acquisition rule is now clearer: keep wins eligible, r8-confirm strong/borderline
  candidates, and do not blindly add flat cheap blocks to ranker training.
- **2026-06-07 win-heavy block check:** offset `720000` captured 18 ante-8 snapshots with wins
  kept, 12 from wins and 6 from near-wins. Cheap labels were still mostly flat: heuristic within
  `0.10` on 15/18 states, mean best-vs-heuristic advantage `+0.065`, mean LCB `+0.011`, and
  practical high-confidence best-vs-heuristic rate `0.056`. The block gate classified it as
  `weak_or_mixed`. The strict selector found one near-win `open_pack` over `end_shop`, but r8
  confirmation made the exact proposal ambiguous with solver LCB `+0.012`. This is good evidence
  that winning trajectory metadata is necessary but not sufficient: keep wins in capture, but only
  spend training/r8/sweep budget when action-level paired signal appears.
- **2026-06-07 quality-filtered checkpoint gate:** trained attention and mean shop-ranker
  checkpoints on the quality-gated pool (`84` examples, strong `700000` cheap block plus all r8
  confirmations, weak/calibration mixed cheap blocks excluded). Training-pool regret looked good
  but was not trusted as deployment evidence. Fresh ante-8 override capture at offset `730000`,
  one ranker action per run and baseline-margin gate `0.25`, produced three `open_pack` proposals
  from attention and three from mean. R8 exact-proposal audits made both checkpoints 0 positive /
  0 negative / 3 ambiguous, with no high-confidence best-vs-heuristic states. No live A/B was run.
  The audit now reports ranker margins; ambiguous proposals still had mean baseline margins
  `0.354` for attention and `0.397` for mean, so model margin is not calibrated yet. Current
  status: use these checkpoints for proposal/acquisition only; require positive r8
  deployment-disagreement labels before testing a shop-ranker bot live.
- **2026-06-08 SHOP path CLOSED — Stage 2 retargeting (reconcile with project memory).** The
  on-policy value-as-shop-leaf experiment (capture 384 solver runs to kill distribution shift;
  attention val win-AUC 0.708 but **std on real solver shop states only 0.053**, flatter than the
  off-policy 0.073) falsifies distribution-shift as the fixable cause: the eventual outcome barely
  depends on shop-choice differences. This subsumes every Stage-2 shop result above — the
  horizon-2 ranker beats the heuristic OFFLINE (regret 0.05 vs 0.10) but is online-flat (best A/Bs
  ~6/5 wins, neutral) for the SAME reason (offline regret rewards matching the rollout-best; online
  it can't move winrate because the choice doesn't move the outcome). **Do not resume the shop
  candidate ranker / shop-leaf / shop-override.** Independent confirmations: shop selection is
  near-optimal (1/30 disagreements); the ante-8 out-test shows 73.7% of losses had an affordable
  offered joker that clears, i.e. the gap is whole-run build CONSTRUCTION reached myopically, not a
  per-shop selection error; META blind-SKIP tested net-negative (21->11/100) with no static
  tag/ante pattern. **Caveat on the 73.7% (measured 2026-06-08):** the out-test GRAFTS the joker
  (adds a 6th, ignoring slots). But late builds are mostly slot-FULL — 9/12 losses reaching an
  ante>=6 shop had 0 free joker slots; 13/20 losses end 5/5 — so a *realizable* late buy needs a
  SWAP (sell to make room), which erases much of the grafted gain (and the bot's shop search already
  searches sells; it just doesn't make the clearing swap because its myopic value can't see the
  clear). So the realizable late-shop headroom is far below 73.7%, and even capturing it points back
  to building stronger EARLIER (before slots fill) = whole-run policy, not a bounded late-shop patch.
  A bounded late-shop *clearing rollout* (short horizon, evaluate buy/swap by actual next-blind
  clear instead of myopic value) is the least-expensive thing to try first IF pursuing this, but the
  slot constraint caps its ceiling. **Net: the value head's useful niche is whole-run state discrimination (a PLAY/
  search leaf), not shop micro-choices.** Retarget Stage 2/3: the first neural+search target is
  WHOLE-RUN FORESIGHT (does this build reach the future wall), evaluated by search over the forward
  model with the value net as the leaf — NOT a per-shop ranker. The hard open question Stage 3 must
  answer first (cheap probe before any big build): *given that shop choices are outcome-flat under
  the current greedy play, what decision surface is NOT flat?* If none is, winrate is play/RNG-
  capped and the lever is the Stage-4 self-play loop changing the POLICY (not just evaluating a
  fixed one). Pin this down before committing to the search engine.
- **2026-06-08 efficiency (enables the search build).** Sim made ~13-15% faster, behavior-identical
  (deployed bot, controlled 16-seed wall 84.8s->72.3s; winrate 22/100 unchanged): hand_draw_odds
  deck-signature memo, best-play tie-break via Rust (score,chips,mult) instead of re-evaluating ~25
  tied subsets/call, and _pool_records shop-availability memo. Plus prior rust-default-ON + ECON_W
  + leaner shop knobs. Every rollout/search the Stage-3/4 engine runs is now correspondingly
  cheaper. See PROGRESS.md + [[project_winrate_status]].
- **2026-06-09 S0 FORESIGHT GATE RUN -> FAILS. Stop signal for the AZ archetype-commitment engine.**
  Ran the decisive go/no-go (the "what surface is non-flat / is commit-foresight learnable" probe).
  Scripts: `scripts/phase8_archetype_oracle.py` (now persists per-seed `rows`), `s0_early_state_capture.py`
  (baseline early-state @ ante 2/3 via a policy-wrapper, so captured state == the labeled trajectory),
  `s0_foresight_classifier.py` (torch multinomial logreg + numpy AUC + offline gap-closure sim).
  Artifacts: `.data/s0_oracle_white_200.json`, `.data/s0_early_features_white_200.json`.
  - Oracle @ 200 held-out white seeds (DEPLOYED basic-play backend) **replicates +6.5%** (baseline
    39/200=19.5% -> best-of-4 52/200=26.0%) -- ceiling real but **flush-dominated** (best=flush 72 /
    baseline 124 / scaling+high_card+pair ~1).
  - Deck-suit-concentration (the hypothesized predictor) is NULL: std 0.001 through ante 3 (baseline
    never stacks suits). "flush helps this seed" held-out AUC = **0.41-0.51** across logreg+MLP and
    ante 2/3/combined = chance. Conservative selector captures **0% to negative** of the gap.
  - Root cause: the basin that wins is set by the run's LATER RNG offers, unobservable at commit ->
    a reactive policy/value head CANNOT learn the commitment (AUC at chance), and even a perfect
    forward-search oracle caps at ~26% (not superhuman). **Decision: FAIL -> do not build the engine
    on the archetype lever** (steps S1-S5 of the proposed path are not worth funding). ~22-26% is
    at/near the white-stake ceiling for this heuristic architecture. The path's own week-1 kill-switch
    did its job: a quarter's build avoided for ~1.5 hr of compute. See PROGRESS.md 2026-06-09.
