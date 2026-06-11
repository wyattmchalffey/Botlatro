# Phase 8 Current Status

> **SUPERSEDED 2026-06-09:** the shop-ranker program below is closed (per-decision shop work is a
> measured dead end), and the "Next Work" list no longer reflects priorities. The current plan is
> `SUPERHUMAN_ROADMAP.md` (repo root): P0 fix instruments (incl. the known_deck foresight leak) ->
> P1 whole-run planner -> P2 self-play value iteration. This file is kept as the shop-ranker
> post-mortem record.

Last updated after the quality-filtered ranker override gate.

## Short Version

We are making real progress on the neural shop-ranker path, but we are not at a deployable neural
shop bot yet. The strongest result so far is not a live winrate improvement; it is that we now have
a better label acquisition funnel:

- keep both wins and near-wins in late-shop capture,
- audit each cheap label block before training on it,
- spend expensive r8 confirmation only on action-level disagreements,
- use rankers as proposal models until fresh override labels are positive, not merely ambiguous.

The current quality-filtered ranker checkpoints should be treated as acquisition/proposal tools,
not as live shop overrides.

## What Improved

- The Rust best-play path is hooked into the labeler path and reports `rust_bestplay: true` in the
  relevant metrics.
- Shop candidate ranking is the most promising neural formulation so far. Whole-action candidates
  work much better than trying to predict per-card or long-horizon scalar value directly.
- Late-shop backward capture now includes winning trajectories instead of filtering all wins away.
  This matters because some final shops are what converted a run.
- Relabeled records now preserve terminal metadata such as `terminal_won`, `selection_reason`,
  terminal score/money, and shops-from-terminal.
- The deepening selector can balance r8 confirmations across win and near-win terminal outcomes.
- `phase8_shop_confidence_audit.py` now classifies cheap label blocks as `strong_signal`,
  `weak_or_mixed`, or `calibration_only`.
- `phase8_deepening_confirmation_audit.py` now reports ranker margins so we can see when model
  confidence fails to transfer to solver confirmation.

## Best Signal So Far

The best cheap block so far is offset `700000`:

- 24 ante-8 mixed late-shop snapshots.
- 6 records from wins and 18 from near-wins.
- r4 cheap labels were signal-rich: mean best-vs-heuristic advantage `+0.249`, mean LCB `+0.092`.
- Strict selector found 5 fresh opportunities, balanced across 2 wins and 3 near-wins.
- r8 exact proposal audit: 4 positive / 0 negative / 1 ambiguous.

With r8 confirmations overlaid, the 122-state sweep was the strongest all-in read:

- 37 positive / 51 negative / 376 ambiguous candidate labels.
- Mean encoder regret `0.123` vs heuristic `0.156`.
- Attention encoder regret `0.119` vs heuristic `0.156`.
- Both encoders beat heuristic regret on all 7 split seeds.
- Mean encoder was safer on calibration, but threshold `0.1` still covered harmful overrides at
  `0.086`, so it was not deployable.

## Quality-Filtered Gate

The block audit correctly separated useful signal from noise:

- `680000`: `weak_or_mixed`
- `690000`: `weak_or_mixed`
- `700000`: `strong_signal`
- `710000`: `calibration_only`
- `720000`: `weak_or_mixed`

Filtering training to the strong `700000` cheap block plus all r8 confirmations produced a denser
84-example pool:

- 37 positive / 28 negative / 252 ambiguous labels in the 83-example pre-final sweep.
- Mean regret `0.135` vs heuristic `0.200` across the filtered split sweep.
- Attention regret `0.132` vs heuristic `0.200`.
- Both won 6/7 splits.
- Attention had the best calibrated read in that gate: calibrated lift `+0.012`, harmful-covered
  rate `0.027`.

This is a useful training hygiene result, not deployment evidence.

## Current Checkpoints

Quality-filtered ranker checkpoints:

- `.data/phase8_shop_ranker_quality_filtered_attention_v1.pt`
- `.data/phase8_shop_ranker_quality_filtered_mean_v1.pt`

Training-pool metrics looked strong:

- Attention final regret `0.087` vs heuristic `0.185`.
- Mean final regret `0.077` vs heuristic `0.185`.

Fresh override capture at offset `730000` did not pass:

- Attention made 3 ante-8 `open_pack` proposals.
- Mean made 3 ante-8 `open_pack` proposals.
- r8 exact proposal audit was 0 positive / 0 negative / 3 ambiguous for both.
- Mean ranker baseline margin was still high on ambiguous proposals: `0.354` attention,
  `0.397` mean.

Conclusion: model margin is not calibrated enough for live overrides.

## Action-Family Separation: buy/card (2026-06-07)

The 730000 "100% open_pack" read was an artifact of ante-8-only + a 0.25 gate. A held-out no-rollout
scoring sweep (offset `740000`, antes 2-8) shows the attention checkpoint proposes 69% open_pack but
also a frequent buy/card family with *higher* mean confidence (margin `0.254` vs open_pack `0.181`).
open_pack is also nearly unconfirmable at r8 because pack RNG widens the CIs.

A buy/card-only fresh capture (offset `750000`, antes 4-6, gate `0.25`) with the 16 most-confident
proposals r8-confirmed to terminal gave the first confirmable positives:

- Exact proposal audit: **5 positive / 4 negative / 7 ambiguous**.
- Positives are real and large: mean advantage `+0.6`..`+1.6`, LCB `+0.17`..`+0.35`.
- Calibration is still broken: positive vs ambiguous ranker margins are indistinguishable
  (`0.530` vs `0.523`); some high-margin proposals are strongly negative (LCB to `-1.08`).
- Mean realized advantage across all 16 most-confident card-buys is only `+0.065` (within noise), so a
  self-gated override would still net ~neutral.
- Clean anti-pattern: overriding `end_shop` with a card-buy was negative 2/2.

Block saved: `.data/phase8_deploy_disagree_block_buycard_750000_r8term.jsonl` (16 records) plus
`..._POS5.jsonl` (5 positives). Net read: action-family separation is the right move for *acquiring*
confirmable labels, but the deployment blocker is now precisely **uncalibrated confidence**, which is a
data-scale problem, not a pipeline bug.

## What Is Holding Us Back

- Label ambiguity: most shop choices are not clearly better or worse under limited rollout budget.
- Multiple viable early paths: there often is no single best action, so one-winner labels are a
  poor fit.
- Long-horizon credit assignment: shop value depends on future economy, scaling, packs, jokers,
  and survival several antes later.
- Distribution shift: a ranker trained on late ante-8 disagreements does not automatically become
  reliable earlier.
- Calibration failure: high model margin can still produce solver-ambiguous proposals.
- Data scarcity: the quality-filtered set is around 84 examples, which is enough for diagnostics
  but not enough for robust deployment.
- Expensive confirmation: r8 labels are slow, so blind scaling wastes time unless cheap blocks pass
  the quality gate.

## Next Work

1. Keep the current rankers as proposal models, not deployed overrides.
2. Capture fresh ranker-vs-baseline disagreements on held-out seeds.
3. r8-confirm those exact disagreements before any live A/B.
4. Treat positive r8 confirmations as the next training set, and treat ambiguous/flat blocks as
   calibration or holdout.
5. Separate action families: `open_pack`/build-forward choices are not the same as `end_shop`,
   skip, sell, or economy-preserving choices.
6. Add economy-aware labels where two lines both survive but one leaves much better money/scaling
   potential.
7. Only run paired unseeded A/B after the r8 override gate produces positive deployment
   disagreements on fresh seeds.

## Cleanup Performed

- Removed completed `.partial` artifact companions whose final files already existed.
- Removed non-venv Python `__pycache__` directories.
- Removed `.pytest_cache`.

