# SearchBot v1 early-loss audit

Date: 2026-05-04

## Seeds audited

High-gap early losses were selected from `search_bot_v1_action_pressure_200.jsonl` by looking for v1 deaths in ante 1-3, plus seeds where Basic reached much later.

Most informative seeds:

- `687914451`: v1 died Ante 1 Small Blind at `296/300`; Basic reached Ante 7.
- `263096377`: v1 died Ante 1 Small Blind at `244/300`; Basic reached Ante 4.
- `919251017`: v1 died Ante 2 Big Blind with Red Card line; Red Card fix turned this seed into a win.
- `959680864`, `1004575915`, `1870520371`: still early deaths after Red Card fix; mostly pack/shop pressure and underbuilt scoring capacity.

## Finding 1: opening discard search can misvalue ties

The Ante 1 deaths were caused by discard-search overrides on the opening no-joker blind. The search action changed the draw path and missed the blind by tiny margins, while Basic preserved a stronger straight/flush or multi-pair line.

A broad guard was tested: "on Ante 1 Small Blind with no jokers, keep Basic on estimated-hand ties." It rescued the two audited seeds:

- `687914451`: Ante 1 Small Blind death -> Ante 8 Crimson Heart
- `263096377`: Ante 1 Small Blind death -> Ante 6 Psychic

But the broad guard was not kept. At 200 seeds it lowered win rate from `5.0%` to `3.5%`, with essentially no ante benefit. It fixed some seeds and hurt others because tiny opening discard changes alter the whole deck path.

Conclusion: this is a real discard value-function weakness, but the fix should be better discard leaf calibration, not a hard opening-blind guard.

## Finding 2: Red Card was overvalued unless the bot actually skips packs

The bot valued unscaled Red Card like a scaling joker even when it did not plan to skip booster picks. That made lines like "buy Red Card, then take pack cards anyway" look much better than they were.

Implemented fixes:

- Unscaled `Red Card` now scores low unless there is a visible booster skip plan.
- Once Red Card has current mult, it regains mult/scaling role value.
- `pack_search` now gives skip actions Red Card's skip value instead of treating skip as zero.

Targeted result:

- `919251017`: Ante 2 Big Blind death -> win.

## Current kept result

Current source keeps the Red Card fix and does not keep the broad opening discard guard.

200-seed local-sim result:

| Variant | Wins | Avg Ante | Avg Score | Avg Money | Small Blind Deaths | Big Blind Deaths |
|---|---:|---:|---:|---:|---:|---:|
| Saved Basic baseline | 10/200 | 5.04 | 18523.2 | 39.90 | 8 | 51 |
| Prior v1 action-pressure | 10/200 | 5.12 | 16741.4 | 26.64 | 20 | 59 |
| Current v1 Red Card fix | 11/200 | 5.22 | 16824.2 | 27.90 | 19 | 63 |

Paired current v1 vs prior action-pressure:

- Win delta: `+0.5%`
- Wins flipped/lost: `2 / 1`
- Average ante delta: `+0.10`
- Wilcoxon ante p-value: `0.029`
- Average score delta: `+82.8`

Paired current v1 vs saved Basic baseline:

- Win delta: `+0.5%`
- Wins flipped/lost: `11 / 10`
- Average ante delta: `+0.18`
- Average score delta: `-1699.0`

## Remaining likely issues

The current bot reaches later antes more often, but Big Blind deaths rose to `63/200`. That points to capacity and money conversion, not just early survival.

Likely next fixes:

- Integrate pack choice into shop beam instead of treating `OPEN_PACK` as terminal.
- Improve discard leaf calibration rather than adding a broad opening-blind guard.
- Re-run the Basic baseline after Red Card valuation changes, because Basic shares that valuation code.
