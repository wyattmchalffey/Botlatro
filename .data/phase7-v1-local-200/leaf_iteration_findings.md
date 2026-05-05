# Phase 7 shop leaf iteration findings

Date: 2026-05-04

## Current kept variant

The current working tree keeps:

- default leaf term tracing with `ShopLeafTerms`
- money-floor penalty from the prior iteration
- capacity headroom value
- pressure-aware owned-joker value
- unresolved-pressure penalty for joker buys that spend money without improving sampled scoring capacity

The pack pressure penalty was tested and reverted.

## Checks

- Focused tests after current kept variant: `36 passed`
- Full suite after current kept variant: `543 passed, 24 subtests passed`

## 200-seed local-sim results

| Variant | Wins | Avg Ante | Avg Score | Avg Money | Small Blind Deaths | Big Blind Deaths |
|---|---:|---:|---:|---:|---:|---:|
| Basic baseline | 10/200 | 5.04 | 18523.2 | 39.90 | 8 | 51 |
| Headroom | 9/200 | 5.03 | 16285.1 | 24.14 | 22 | 55 |
| Owned pressure | 9/200 | 5.13 | 17357.0 | 26.84 | 24 | 45 |
| Current kept: action pressure | 10/200 | 5.12 | 16741.4 | 26.64 | 20 | 59 |

## Paired comparisons

Current kept variant vs Basic baseline:

- Win delta: `+0.0%`
- Wins flipped/lost: `10 / 10`
- Average ante delta: `+0.08`
- Wilcoxon ante p-value: `0.638`
- Average score delta: `-1781.8`
- Score bootstrap 95% CI: `[-6386.2, +2591.3]`

Current kept variant vs owned-pressure-only:

- Win delta: `+0.5%`
- Wins flipped/lost: `2 / 1`
- Average ante delta: `-0.01`
- Average score delta: `-615.5`

## What helped

Pressure-aware owned value was the cleanest improvement. It reduced the "inventory value dominates everything" problem seen in traces, improved money, improved average ante versus headroom, and reduced Big Blind deaths from `55` to `45`.

The joker-buy unresolved-pressure guard fixed a concrete trace miss: seed `959680864` no longer bought an inactive `Driver's License` while still having a pressure ratio near `3.9` and no sampled score improvement. It brought wins back to Basic's `10/200` and reduced Small Blind deaths from `24` to `20`, but shifted some losses into Big Blind and lowered score.

## What did not help

The pack-pressure penalty was too blunt. At 50 seeds it dropped average ante from `5.32` to `5.00` versus the action-pressure variant and dropped score by about `3974`. It was reverted.

This suggests pack opens are still important even in ugly shops, and the right fix is probably deeper pack lookahead or better pack-choice leaf scoring, not a generic penalty before opening.

## Current diagnosis

SearchBot v1 is no longer clearly worse than Basic in local sim, but it is not yet clearly better either. The remaining problem is not just "spends too much"; it is spending differently and converting less of that spending into score and bank.

The next best target is pack/shop sequence lookahead:

- evaluate visible pack contents after opening instead of treating `OPEN_PACK` as terminal
- carry pack-choice value back into the shop beam
- keep the current term tracing so we can see whether packs improve `build_score`, money, or role completion

That should attack the current failure mode without blocking packs blindly.

## Interest-cap floor update

The shop money floor now uses `$5` interest breakpoints:

- no voucher: `$5`, `$10`, `$15`, `$20`, then `$25` in later antes
- Seed Money: ramps above `$25` toward the real `$50` interest cap
- Money Tree: ramps above `$25` toward the real `$100` interest cap

The safe-shop adjustment also preserves raised voucher caps instead of snapping the floor back to `$25`.

Checks after the update:

- `tests/test_shop_search.py`: `28 passed`
- `tests/test_basic_strategy_bot.py tests/test_search_bot.py`: `109 passed`
- full suite: `549 passed, 24 subtests passed`

50-seed local-sim smoke, current code:

- file: `.data/phase7-v1-local-200/search_bot_v1_interest_cap_floor_fixed_50.jsonl`
- wins: `1/50`
- average ante: `5.32`
- average score: `18733.9`
- average money: `27.3`

Paired against the prior breakpoint-only floor on the same 50 seeds:

- win delta: `+0.0%`
- average ante delta: `+0.06`
- average score delta: `+1114.0`
- score 95% CI: `[-275.0, +2871.9]`

200-seed local-sim comparison of the raised-cap floor:

- file: `.data/phase7-v1-local-200/search_bot_v1_interest_cap_floor_fixed_200.jsonl`
- wins: `10/200`
- average ante: `5.09`
- average score: `15956.6`
- average money: `29.2`

Paired against prior current-best `search_bot_v1_red_card_no_opening_guard_200`:

- win delta: `-0.5%`
- wins flipped/lost: `2 / 3`
- average ante delta: `-0.12`
- Wilcoxon ante p-value: `0.041905`
- average score delta: `-867.6`
- score 95% CI: `[-2672.1, +775.3]`

Paired against Basic baseline:

- win delta: `+0.0%`
- wins flipped/lost: `10 / 10`
- average ante delta: `+0.06`
- Wilcoxon ante p-value: `0.610017`
- average score delta: `-2566.6`
- score 95% CI: `[-7079.8, +1548.5]`

Read: the raised-cap floor is correct for Seed Money/Money Tree modeling, but the stricter ordinary money floor is too conservative as a default. It should not be promoted as the current performance variant without softening the non-voucher floor or gating the higher target to safe/high-bank situations.

## Economy tracking update

Local-sim run summaries now include an `economy` payload with:

- effective purchase power: starting money plus all money gained during the run
- income buckets: cash-out, sells, consumables, discards, play, skip, other
- spend buckets: jokers, consumables, booster packs, vouchers, rerolls, playing cards, blind effects, other
- action counts for buys, packs, rerolls, boss rerolls, and sells

`local_benchmark.py` prints average income, average spend, effective money, and spend by type when economy data is present. `compare.py` now prints paired economy deltas when both result files include the payload.

Smoke check:

- file: `.data/phase7-v1-local-200/economy_tracking_smoke_5.jsonl`
- average effective money: `85.8`
- average income: `81.8`
- average spent: `45.8`
- average spend by type: jokers=`27.0`, consumables=`1.2`, packs=`9.2`, vouchers=`0.0`, rerolls=`8.4`, cards=`0.0`, other=`0.0`

## 200-seed economy rerun

Files:

- Basic: `.data/phase7-v1-local-200/basic_strategy_economy_200.jsonl`
- Search v1: `.data/phase7-v1-local-200/search_bot_v1_economy_200.jsonl`
- Comparison: `.data/phase7-v1-local-200/comparison_search_v1_economy_200_vs_basic.json`

Run results:

| Variant | Wins | Avg Ante | Avg Score | Avg Final Money | Effective Money | Income | Spent |
|---|---:|---:|---:|---:|---:|---:|---:|
| Basic | 10/200 | 5.07 | 18388.3 | 39.6 | 145.8 | 141.8 | 106.3 |
| Search v1 | 9/200 | 5.09 | 16111.9 | 29.4 | 155.1 | 151.1 | 125.7 |

Paired deltas, Search v1 minus Basic:

- win delta: `-0.5%`
- average ante delta: `+0.03`
- average score delta: `-2276.4`
- effective purchase power delta: `+9.2`
- income delta: `+9.2`
- spent delta: `+19.4`
- final money delta: `-10.2`

Spend deltas, Search v1 minus Basic:

- jokers: `+1.2`
- consumables: `+1.5`
- packs: `+18.2`
- vouchers: `+14.6`
- rerolls: `-15.0`
- playing cards: `+0.2`
- blind effects: `-1.3`

Read: Search v1 is not starved for effective purchasing power. It earns more than Basic, spends much more, and ends poorer. The biggest behavioral shift is overbuying packs/vouchers while under-rerolling. The next shop fix should audit pack/voucher value and opportunity cost rather than simply raising the money floor.
