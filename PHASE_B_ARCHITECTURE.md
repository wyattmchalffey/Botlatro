# Phase B: Decision-Shaped Policy — the chassis replacement

*2026-06-12. The design for the program that replaces `basic_strategy` as the
deployed policy. Written after the week's verdicts; every choice below cites
the measured result that forces it. Companion docs: P2_EVAL_PROTOCOL.md
(seed discipline, gates), P2_COMPUTE_DECISION.md (rent, ~$15-50/iteration).*

## Why a replacement, not another augmentation

The bolt-on record is now 0-for-everything: value-as-shop-leaf (catastrophic),
policy-pruned beam (dead end), ranker overrides (uncalibrated), deep-play
delegation (-5.9pp, flip-diag: the beam is better than basic ONLY conditioned
on basic failing), honest draw probabilities (neutral). The recurring
mechanism is co-design: basic_strategy's components are mutually adapted, and
any foreign decision-maker spliced into one surface degrades the whole. The
only configuration that escapes this is one policy making ALL decisions —
trained, not hand-built.

## The one architecture lesson that worked

Stage 2.2's candidate-subset play policy is the single Phase-8 head that beat
its baselines (top-1 0.388 vs 0.031 random, generalized across seeds; 0.53
post encoder fixes): **don't predict from the pooled state — score the
enumerated candidate actions directly**, each candidate encoded with explicit
per-candidate features plus shared state context. Its negative results are
equally load-bearing: per-card pointers ~chance, hand-type heads ~chance,
pooled scalar values flat (std 0.05 on real decisions). Decision-shaped or
nothing.

## Network

One shared trunk, one candidate-scoring mechanism, every decision type.

- **State trunk**: the existing token encoder (`ml/model.py` ValueNet:
  joker/card/shop token encoders + attention + trunk), proven order-blind
  (231-state probe). Scaled UP: Phase 8 ran ~50k params with d_token=32 —
  model capacity was never the tested variable. Start d_token=128,
  d_trunk=256, 4 attention layers (~2-4M params; inference still sub-ms on
  CPU per decision batch).
- **Candidate tokens**: every legal action becomes a candidate with
  type-specific features, INCLUDING THE HEURISTIC'S OWN EVALUATIONS as
  inputs — feature-level fusion instead of policy-level overrides:
  - play/discard: card-subset pooled embedding, hand type, size, Rust batch
    immediate score + boss-adjusted score, draw-odds summary for the kept
    hand (`hand_draw_odds` probabilities), discards/hands remaining.
  - shop buy/sell/reroll/end: item token (existing shop-token encoder),
    cost/money-after, `shop_leaf_terms`-style value components as features,
    slot state.
  - blind select/skip: tag identity + the would-be-shop sampler summary,
    blind chip requirement vs build projection.
  - pack pick / consumable use: card token + same fusion pattern.
  The net's job is learning WHEN the heuristic's evaluation is wrong over
  the whole run — not rediscovering chip math the Rust scorer computes
  exactly. This is the structural answer to co-design: the heuristic
  becomes vocabulary, not chassis.
- **Heads**: (a) policy: per-candidate scalar -> softmax over legal
  candidates (negative-sampling CE at train time, full enumeration at
  inference); (b) value: P(win) + ante-reached auxiliary off the trunk CLS
  (kept for Phase-C search hooks and for diagnostics — NOT deployed as an
  argmax-override; that pattern is dead).

## Training program

- **Iteration 0 — behavior cloning on the diverse mixture.** Generate ~50k
  honest-mode runs (shuffle), seeds 5,100,000+, rented (~$30): mixture =
  deployed bot (anchor, ~60%) + the 7 `recipe_*_bot` wrappers (~40% — the
  S0 antidote: the data must CONTAIN suit-stacking, skip-routes, leveling
  pushes for any of it to be learnable). Train policy on taken actions,
  value on outcomes. GATE B0: the BC policy, deployed as `neural_policy_bot`
  (registry bot; encoder + net for every phase; fall back to basic ONLY on
  illegal/errored output), must come within 3pp of the deployed bot on a
  512-seed holdout slice. This gate proves the action space, encoder, and
  deployment plumbing carry a whole policy before any improvement claims.
- **Iteration 1+ — improvement loop.** Generate from the current neural
  policy with exploration (temperature sampling on the policy head +
  continued recipe-mixture episodes ~20%), retrain (policy: CE toward
  actions of WINNING trajectories upweighted / advantage-weighted by the
  value head; value: TD(lambda) + target network — the two fixes that each
  worked and were never combined). Evaluate per P2_EVAL_PROTOCOL: paired
  512 holdout, adopt iff McNemar p<0.05 positive; design review after 2
  flat iterations.
- **Phase-C hook (not now)**: if/when iterations plateau, the value head +
  candidate scores feed a shallow chance-aware search (the 2048/MuZero
  formula). The architecture is chosen so this bolt-ON is actually a
  bolt-IN: same net, more lookahead.

## What this program explicitly does not do (the graveyard clauses)

- No confidence-gated overrides of the heuristic (calibration is the
  recurring corpse: ranker margins 0.53 pos vs 0.52 ambiguous).
- No clairvoyant or hindsight-relabeled targets (realizability; the
  rollout-relabel dead end). Labels are honest outcomes of honest play.
- No shop-only or surface-only heads. Whole-policy or nothing.
- No training on pre-foresight-fix data; everything regenerates under
  shuffle mode with the order-blind encoder.
- No silent metric upgrades: every iteration reports the pre-registered
  gate numbers, win/loss flips, and CI.

## Prerequisites before iteration-0 generation spends money

1. Late-ante RNG certification (P0.4 bridge session — tooling ready).
2. Data-gen cost gates (beam_depth 3->2, SHOP_BEAM_WIDTH=2 for bulk
   generation — pre-vetted, quality-gated; ~10-30% cheaper runs).
3. Encoder candidate-feature extension (the per-candidate fusion features
   above are NEW encoder work — the one substantive engineering chunk in
   this doc; estimate 3-5 days including tests).
4. Clairvoyant ceiling number (defines what the program is aiming at).

## Honest expectations

BC iteration-0 lands near the mixture's winrate (~10-13%) by construction.
The bet is that iterations 1-3 climb meaningfully past 12.4% because (a) the
policy can deviate from the heuristic EVERYWHERE simultaneously (no co-design
wall), (b) the data contains forced diversity the baseline never produced,
and (c) outcome labels at 50k-run scale attack the run-starvation that capped
every Phase-8 head (train-AUC 0.999 / val 0.708 at 384 runs). If two
iterations go flat, the design review happens BEFORE Phase-C search is
invoked as a rescue — search amplifies a good value function, it does not
repair a bad one (shop-depth lesson).
