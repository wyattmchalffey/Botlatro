# Search Bot V2 Ideas

Starting point: `search_bot_v2` is a separate experiment lane layered over
`basic_strategy_bot`. It calls search modules for shop, pack, consumable, and
hand decisions, while keeping Basic Strategy as the fallback and as a guard for
some play/discard decisions.

Current anchor: `basic_strategy_bot` is the confirmed baseline. On the strict
200-seed local-sim set it is confirmed at `23/200` White Stake wins (`11.5%`)
after the broad all-hand discard evaluator was reverted. `search_bot_v2` should
not be treated as the active baseline until it beats that same-seed result.

Recent signal:

- Current confirmed Basic Strategy strict set: `23/200` wins, `11.5%`.
- The broad all-hand discard evaluator was tested and reverted after a clear
  regression. Do not reintroduce broad discard rewrites without a tight gate and
  same-seed A/B.
- `search_bot_v2` currently has no confirmed 200-seed result beating Basic
  Strategy. Small samples are too noisy and the search path can be much slower.
- Older live-bridge v1 numbers below are historical debugging context, not the
  current winrate baseline.

## Highest Priority

1. Make v2 prove itself against the current Basic baseline.
   - Run same-seed local-sim A/Bs against `.data/codex-revert-confirm-strict200.jsonl` seeds.
   - Keep changes only if they improve wins or average ante without large runtime regressions.
   - Track flips against Basic Strategy, not only aggregate winrate.

2. Keep Basic Strategy as the safety guard.
   - V2 should continue to ask Basic Strategy for blind actions and only
     override when the search result has a clear modeled edge.
   - Be especially careful with play-vs-discard flips, which can erase
     hard-won early-ante survival.

3. Prefer narrow, auditable experiments.
   - Good candidates: shop budget gates, pack ordering when buying/opening both
     actions is already planned, replacement-sell safety, and high-pressure
     reroll valuation.
   - Avoid broad "evaluate everything" rewrites until each hand family has
     focused tests and same-seed evidence.

## Candidate Experiments

1. Add a shop-search safety budget tied to clear capacity.
   - Try reducing speculative pack/reroll/buy value when the build is not above
     the next-blind clear bar.
   - Keep this narrow: gate spending and compare same-seed flips.

2. Improve early blind play/discard safety.
   - Several current losses are Ante 1-2, where shop search cannot help yet.
   - Audit early losses for discard lines that chase low-probability hands instead of banking safe clears.
   - Expand known-discard safety from last-discard cases into low-hand-count, low-score states if the draw model is overconfident.

3. Make replacement sells require real build improvement.
   - Revisit the simple replacement gate, but apply it only when selling a scoring joker for cash or a low-impact visible replacement.
   - Avoid the previous broad unresolved-sell penalty; focus on direct sell-to-buy/sell-to-end lines.

4. Improve pack choice quality instead of blanket pack avoidance.
   - Classify pack value by current missing role and slot pressure.
   - Celestial packs should matter when a real hand-plan exists, but should be weaker when open joker slots and missing joker roles are the immediate bottleneck.

5. Add same-seed canary command for v2.
   - Use the strict 200-seed Basic baseline plus any gained/lost flip seeds from
     a recent A/B.
   - A v2 change should pass legality, avoid runtime blowups, and improve net
     flips before a larger smoke.

## Suggested First V2 Branch

Start with a same-seed A/B harness around the current strict 200-seed Basic
baseline. Then choose one narrow shop-budget or sell-replacement change and
measure net flips. Do not chase Phase 8 until the rule/search teacher is much
closer to `40%` to `50%+` White Stake winrate.
