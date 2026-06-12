# Phase B: Decision-Shaped Policy — the chassis replacement (v2)

*2026-06-12. v2 after the 3-lens adversarial review (graveyard / feasibility /
learning-theory, all `sound-with-fixes`; verdicts in the session record). Every
v1 blocking issue is addressed inline and marked `[R]`. Companion docs:
P2_EVAL_PROTOCOL.md, P2_COMPUTE_DECISION.md.*

## Why a replacement, not another augmentation

The bolt-on record: value-as-shop-leaf (catastrophic), policy-pruned beam
(dead end), ranker overrides (uncalibrated), deep-play delegation (-5.9pp;
flip-diag: the beam beats basic only conditioned on basic failing), honest
draw probabilities (neutral). The shared mechanism is co-design — but `[R]`
the graveyard's proximate causes do NOT all die with the chassis, so each
must be neutralized explicitly:

- **Uncalibrated confidence** killed overrides → here there are no overrides:
  the policy head IS the decision (softmax argmax), never a gated veto of
  another decider.
- **Flat values** killed value-argmax → here the value head is never an
  action selector: state-level baseline + diagnostics only. `[R]` The
  2026-06-08 on-policy test FALSIFIED "more data fixes flat shop values"
  (std 0.053 at val-AUC 0.708, flatter than off-policy; "eventual outcome
  barely depends on shop-choice differences" is a domain property). A priori,
  expect ~nil improvement signal at shop surfaces; the bet lives on play,
  routing, and pack/consumable surfaces.
- **Imitation compression** killed the play policy deployed as a pruner
  (rank-corr -0.68 with its own teacher) → here BC is explicitly only the
  bootstrap, the loop is the product, and the data must contain non-teacher
  behavior on EVERY surface (see Diversity, `[R]` rewritten).

## The architecture lesson (stated honestly)

`[R]` The candidate-subset head is the strongest Phase-8 OFFLINE result
(top-1 0.388 vs 0.031 random; train≈val so NOT run-starved), but its one
deployment test failed (Stage 2.3 beam pruning, -1.4 antes) — the lesson
"score enumerated candidates with explicit features" is real; its deployment
record is 0-for-1 and the fusion design below is untested. The action-type
head (0.75 vs 0.23) and distilled clear-leaf (corr 0.87) also beat baselines;
pooled scalar values and per-card pointers did not. Decision-shaped is the
best-supported direction, not a proven one.

## Network

- **Trunk**: existing token encoder (ml/model.py), proven order-blind
  (231-state probe). `[R]` Measured on this box (Ryzen 3800X, CPU torch):
  d_token=128 / d_trunk=256 / 4 attention layers = **0.97M params**, forward
  **~3-4ms** per decision with 218 candidates — fine for benches (~1.5x
  current play-decision cost), not "sub-ms".
- **`[R]` Encoder gaps to close first**: BOOSTER_OPENED pack contents are
  INVISIBLE to encode_state today (only shop-offer tokens exist), and there
  is no tag vocabulary — both are required for whole-policy deployment, not
  optional polish.
- **Candidate tokens with heuristic fusion**: every legal action becomes a
  candidate carrying type-specific features INCLUDING the heuristic's own
  evaluations (Rust batch immediate/boss-adjusted score, draw-odds summary,
  shop value terms, tag identity + would-be-shop summary). The net learns
  when the heuristic is wrong over the run, not chip math. `[R]` Two
  mitigations from review: (a) missing-feature indicators — the Rust scores
  bail to None on exactly the hard boss states (RUST_BLIND_SAFE), so absence
  is informative and must be encoded, not zero-filled; (b) anti-shortcut:
  feature-dropout on the heuristic-score features during BC + a reported
  trunk-only ablation, so the net cannot pass gates by learning
  argmax-of-the-feature (the training-time analog of Stage 2.3).
- **Heads**: per-candidate scalar → softmax (negative-sampling CE at train,
  full enumeration at inference); value = P(win) + ante-reached auxiliary,
  baseline/diagnostic only.

## Diversity (rewritten — the v1 design failed its own argument) `[R]`

v1's recipe mixture had NO suit-stacking recipe (the one dimension S0 proved
valuable and unlearnable-without-data) and ZERO play diversity (recipes
override only shop/blind-select; 100% of play labels were the teacher's —
imitation-only on the surface holding the largest measured headroom, 29.7%).
Iteration-0 data therefore adds:

1. **A flush-commit recipe** (suit-stacking): shop+play biased toward a
   target suit (the `solver/archetypes.py` machinery exists; wrap it as
   `recipe_flush_commit_bot`).
2. **Play-diversity episodes**: a fraction of episodes play with
   candidate-level perturbation (sample from the heuristic's top-k play
   candidates instead of argmax — the `_play_candidates` list already
   exists), so play labels contain ranked alternatives, not one teacher line.
3. **Dense play labels from search** `[R]` (review's best suggestion): the
   fork-audit machinery (`endgame_play_audit`) becomes a LABEL SOURCE — at
   death blinds, the d6w6 beam's clearing lines (29.7% of losses have one)
   are search-improved targets on exactly the surface where credit
   assignment is tightest. This is the "unanimous formula" (search-improved
   targets) at day-1 cost, not Phase C.

## Training program

- **Pre-step (cheap, before any spend)** `[R]`: bench every recipe bot
  standalone (~128 seeds each) — the mixture's winrate is currently
  UNMEASURED and is both the B0 reference and the winning-mass input below.
- **Iteration 0 — BC on the mixture** (~50k honest runs, seeds 5.1M+,
  rented): deployed ~55% + recipes ~35% + play-perturbation ~10%.
  `[R]` Winning-mass arithmetic the v1 omitted: at ~12% mixture winrate this
  yields only ~5-6k winning trajectories (~100-300 per forced strategy).
  Therefore iteration 0 trains BC on ALL trajectories (diversity preserved);
  outcome weighting enters only at iteration 1+, and as a TILT
  (advantage-weighted with per-policy baselines), never a filter — naive
  winner-upweighting would delete the recipe diversity it rides on.
- **GATE B0 (re-scoped)** `[R]`: a PLUMBING gate, stated as such — the BC
  policy deployed as `neural_policy_bot` must land within a pre-registered
  CI of the MEASURED MIXTURE winrate (not the deployed bot's), evaluated on
  TRAINING-RANGE seeds (it consumes no reserved holdout slice). It certifies
  encoder/action-space/deployment plumbing only.
- **GATE V0 (new)** `[R]`: before any advantage-weighted iteration, the
  value head must demonstrate per-decision advantage RESOLUTION on held-out
  states (spread + sign agreement with realized outcomes on
  matched-state action pairs) — the component this update leans on is the
  project's 0-for-5 component; it does not get load-bearing status for free.
  If V0 fails: iteration 1 falls back to win-conditioned BC with
  state-level (not decision-level) baselines + the dense play labels.
- **Iterations 1+**: generate from current policy (exploration: temperature
  ONLY on decisions where the policy's top-2 margin is small and the blind
  is not must-clear `[R]` — naive uniform temperature lowers data-gen
  winrate and shrinks the winning mass; ~20% recipe episodes continue),
  retrain, evaluate.
- **Iteration gates (re-powered)** `[R]`: 512-seed gates have MDE ~3-3.7pp
  while realistic per-iteration gains are +1-2pp — the v1 kill clause would
  fire on a genuinely climbing loop. Iteration gates run at **1024-2048
  paired seeds** (~1-2.5h, cheap vs a $30 generation), report d_winrate CI
  AND mean-ante surrogate, and the design-review clause triggers on
  CUMULATIVE non-improvement over 2 iterations (sum-CI excluding +2pp),
  not two individually-insignificant reads.
- **Value training**: `[R]` honest restatement (v1 misquoted the record —
  the target network was NEVER implemented; TD(lambda)'s one win was
  on-distribution only and its deployment test failed): value training is
  MC-outcome first, TD(lambda) as a gated experiment, target-network as the
  textbook stabilizer to TRY against the documented fitted-iteration
  collapse — none of it is "known working"; V0 is its validity gate.

## Engineering inventory (honest, replaces v1's "3-5 days") `[R]`

With the documented 3-5x optimism factor applied — **realistically 2-4 weeks**:
1. Dataset schema v2: per-decision candidate sets + fusion features extracted
   AT CAPTURE TIME (TrainingExample carries none of this today; train-time
   candidates have no GameState access), with policy-provenance +
   temperature fields, sharded pre-tensorized storage (naive full-candidate
   storage is ~140GB vs 57GB free — store top-K candidates + compressed
   features), and per-run incremental writes (the current capture's
   "resumable" cache is all-or-nothing; a crash at run 49k loses everything).
2. Encoder: pack-contents tokens, tag vocabulary, candidate-feature inputs,
   missing-feature indicators.
3. Capture: bot-mixture support (single --bot today), provenance threading.
4. Trainer: multi-decision-type candidate trainer (today: value-only train(),
   play-only play_policy) + the anti-shortcut ablation harness.
5. neural_policy_bot: action construction for every phase + legality
   fallback.

## Prerequisites before iteration-0 spends money

1. Late-ante RNG certification (P0.4 bridge session — tooling ready).
2. Data-gen cost gates (beam_depth 3->2, width 2 for bulk generation).
3. Recipe standalone benches (the pre-step above).
4. Clairvoyant ceiling number (defines the aim point).
5. The engineering inventory above.

## Honest expectations `[R]`

The headroom map this climbs into: the 14.3% route-oracle bounds only 8
static late-fork recipes, not this adaptive whole-run class; the play pool is
29.7% of losses; pack/consumable surfaces are unmeasured. Prior-art honesty:
the 2048 results rode dense per-move reward and deterministic afterstates —
outcome-only credit assignment over 150-250 decisions is materially harder,
which is why the dense play labels (fork-audit) and the V0 gate exist.
Iteration 0 lands near the measured mixture winrate by construction; the bet
is +1-2pp per early iteration, readable only at the re-powered gate sizes.
If two cumulative iterations are flat, the design review happens before
Phase-C search is invoked — search amplifies a good value function, it does
not repair a bad one.
