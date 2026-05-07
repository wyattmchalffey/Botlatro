# Search Bot V2 Ideas

Starting point: `search_bot_v2` is a separate module/class copied from current `search_bot_v1` behavior, with only the emitted bot name changed. The current anchor is the known-discard rollback target, and v2 edits should happen in `src/balatro_ai/bots/search_bot_v2.py` so v1 stays comparable.

Recent signal:

- 50-seed known-discard smoke: `9/50` wins, `18.0%`.
- 200-seed current-search smoke on the first 200 baseline seeds: `9/200` wins, `4.5%`.
- Same first 200 baseline seeds: `basic_strategy_bot` had `10/200` wins, `5.0%`.
- Current search had `13` persistent bridge/action errors after retry.
- On non-error paired seeds, current search averaged `-0.52` ante versus baseline.
- Current search gained `7` wins baseline missed, lost `8` baseline wins, and shared `2` wins.

## Highest Priority

1. Fix bridge/action error paths before tuning winrate.
   - Error seeds include Aura/Trance exact-target failures, invalid `use` state, and card-index drift.
   - Add strict target validation for Spectral/Tarot use actions before returning them from consumable/shop search.
   - Add stale-state recovery or action refresh when bridge-visible pack/consumable state changes underneath the search.

2. Audit lost baseline wins before changing heuristics.
   - Lost baseline-win seeds: `1522815768`, `519744964`, `1625296097`, `93680211`, `745875321`, `176970671`, `864513115`, `1294971832`.
   - Compare first divergent shop/play decision against baseline and current v1.
   - Mark each as search over-spend, wrong sell, wrong pack, wrong discard/play, or bridge/action error.

3. Preserve gained wins.
   - Gained seeds: `1358050382`, `1062307830`, `290344638`, `1665990098`, `922933749`, `1421419275`, `717950687`.
   - Any v2 tuning should run these as a canary so we do not remove the few decisions search is clearly improving.

## Candidate Experiments

1. Add a shop-search safety budget tied to clear capacity.
   - The 200-seed run ended with much lower final money than baseline (`25.7` vs `45.4`).
   - Try reducing speculative pack/reroll/buy value when the build is not above the next-blind clear bar.
   - Keep this narrower than the removed pack/sell penalty work: only gate spending, do not add complex unresolved-sell accounting first.

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
   - Use the first 50 baseline seeds plus gained/lost win seeds.
   - A v2 change should pass legality, preserve most gained wins, and improve at least one lost baseline win before a 200-seed smoke.

## Suggested First V2 Branch

Start with legality/error cleanup, not winrate tuning. A smoke with `13/200` persistent bridge/action errors makes the winrate noisy, and these errors are likely costing real seeds. After that, audit the eight lost baseline wins and choose one narrow shop-budget or sell-replacement change.
