# Botlatro Progress

## Completed

- Created the long-term project plan.
- Created the Python package scaffold.
- Added core action and state models.
- Added deterministic benchmark seed utilities.
- Added benchmark summary metrics.
- Added a random legal-action bot.
- Added the first poker hand evaluator.
- Added a greedy immediate-score bot.
- Added a Gym-like environment wrapper.
- Added JSONL replay logging.
- Added a `run_seed` command for one seeded local bot run.
- Added live execution support to the benchmark command.
- Updated the JSON-RPC client defaults and method mapping for BalatroBot.
- Added BalatroBot API notes from the official docs.
- Added a local preflight command and setup notes.
- Installed uv, Lovely Injector, Steamodded, and BalatroBot locally.
- Patched the local BalatroBot manifest so it loads with the installed Steamodded.
- Launched the BalatroBot bridge and verified `health`.
- Completed a 10-run live `random_bot` smoke benchmark.
- Fixed live action derivation to avoid unaffordable shop buys and impossible boss skips.
- Completed a 100-run live `greedy_bot` benchmark; 3 runs reached ante 2.
- Added `basic_strategy_bot`, which reached ante 2 on 26 of 50 tested White Stake seeds.
- Added a reusable benchmark runner with endpoint-based parallelism.
- Added a Tkinter benchmark GUI with run parameters and worker launch controls.
- Hardened GUI worker launch: workers start sequentially, stale bridges can be stopped first, and partial launches are torn down on failure.
- Added GUI speed presets and changed benchmark defaults to `gamespeed` 32 with true headless mode enabled.
- Fixed the GUI so BalatroBot's mutually-exclusive headless and render-on-API options cannot be enabled together.
- Added a tiny-startup headless Balatro copy path so workers do not flash fullscreen before minimizing.
- Added explicit seed-list support for one-off and hand-picked benchmark runs.
- Added stale-state recovery when BalatroBot rejects an action because the live phase advanced.
- Added score-audit replay metadata and a `balatro_ai.eval.score_audit` command.
- Made benchmark seed failures report as failed runs instead of tearing down the whole benchmark.
- Added benchmark cancellation so GUI stop buttons stop scheduling new seeds instead of producing connection-error runs.
- Extended the score evaluator with suit-debuff bosses, basic enhancements, joker editions, and simple flat/suit joker effects.
- Continued Phase 4 scoring work with 4-worker audits; added The Psychic, Arrowhead, Even Steven, Half Joker, Swashbuckler, Scary Face, and The Flint handling.
- Split score-audit misses into supported versus known-uncertain dynamic cases such as Misprint, Popcorn, Ice Cream, Shoot the Moon, Square Joker, and Ceremonial Dagger.
- Added benchmark metadata for deck, active Balatro profile, and unlock pool; current local default is P1 with all unlocks.
- Extended Phase 4 replay logging with full hand-before-play and held-card details.
- Added evaluator support for debuffed card state, held-card score effects, Blue Joker, Blackboard, Baron, Shoot the Moon, Raised Fist, Steel held cards, Odd Todd, Smiley Face, and several exposed dynamic joker counters.
- Ran a same-seed 4-worker score audit; mean absolute score error improved from 23.5 to 6.3 across 145 played hands.
- Added `balatro_ai.eval.explain_score_misses` to inspect worst replay score misses with hand, held-card, joker-effect, recomputed-score, and suspected-cause context.
- Used the miss explainer to retune Green Joker scoring; current evaluator misses on the latest 8-seed audit are down to 2 supported rows.
- Added first-pass evaluator support for Blueprint, Brainstorm, Four Fingers, Shortcut, Splash, Pareidolia, Hack, Dusk, Seltzer, Sock and Buskin, Hanging Chad, and Photograph.
- Added `balatro_ai.eval.scenario_score` for quick deterministic scorer scenarios without launching Balatro.
- Added `balatro_ai.tools.clean_bridge_logs` and GUI bridge log modes to trim or suppress high-volume Balatro/Lovely bridge log spam.
- Added GUI speed toggles for bridge log mode and replay detail, plus replay modes that can skip full score-audit replay work during large benchmark sweeps.
- Continued Phase 4 joker scoring with Bull, Bootstraps, Acrobat, Seeing Double, Flower Pot, Ancient Joker, The Idol, Triboulet, Baseball Card, Supernova, Ramen, Canio, Yorick, Campfire, and Throwback support.
- Added more metadata-driven scorer support for Steel Joker, Glass Joker, Joker Stencil, Hit the Road, Stone Joker, Castle, Erosion, Driver's License, Loyalty Card, and permanent card chip bonuses.
- Fixed Phase 4 audit misses for Photograph timing, Raised Fist held-card debuff behavior, money-scaled miss explanations, and Baseball Card rarity fallback for Erosion.
- Reran a 20-seed 4-worker White Stake score audit; all 297 played hands matched Balatro exactly, with 0.0 mean absolute error.
- Added a full vanilla joker rarity fallback table from the local Balatro dump and normalized numeric rarity metadata so Baseball Card and future rarity-aware logic work when bridge metadata is sparse.
- Added JSON-backed score scenarios and a `scenario_score --scenario-file` mode for repeatable Phase 4 scoring checks.
- Preserved top-level joker rarity metadata in parsed bridge state.
- Improved `basic_strategy_bot` play/discard discipline: it now prefers the smallest legal hand that clears the current blind pace and scores discard choices using the best potential hand left after discarding.
- Ran a 20-seed live White Stake smoke benchmark after the play/discard update; average ante was 2.00 with 0% win rate, average final money 21.7.
- Added standard-library tests for the foundation.
- Ran larger 100-seed, 4-worker White Stake score-audit smoke tests for `basic_strategy_bot`; the final pass averaged ante 2.08 and reached ante 2+ on 71 of 99 valid replayed runs.
- Fixed deterministic score audit misses found in the larger sweep: Flower Pot now uses scored poker-hand suits including debuffed scored cards but ignoring kicker-only suits, Raised Fist follows Balatro's held-card tie/debuff behavior, Hanging Chad no longer shifts from a debuffed first scoring card, Mad/Clever Joker apply to full houses, and Ramen live text is parsed/classified safely.
- Verified the current evaluator against the final 100-seed replay set: 1,246 supported played hands, 1,246 exact expected-vs-actual score matches; remaining miss rows are known dynamic/uncertain effects such as Misprint, Ice Cream, Bloodstone, Space Joker, Obelisk, The Mouth, The Hook, Green Joker, and Ramen display rounding.
- Started Phase 5 shop/build valuation in `basic_strategy_bot`: shop choices now score jokers, planets, tarots, vouchers, packs, rerolls, edition bonuses, interest breakpoints, simple build synergies, and sample-hand scorer gains instead of buying the first cheap joker.
- Ran a 100-seed, 4-worker White Stake smoke benchmark after the shop update: win rate 1.0%, average ante 3.53, average final score 8,750.5, average final money 45.9, and first observed White Stake win for this bot.
- Added replacement-aware shop play: shop states now expose sell actions, the bot can sell its weakest joker for a major visible upgrade, and shop/pack actions carry reason metadata into replay logs.
- Tightened hand play around hands remaining: the bot now chooses plays by estimated hands-to-clear and only discards when the current hand cannot clear before using all remaining paid hands.
- Ran a 100-seed replay-producing smoke after replacement/min-hands changes; the command timed out at the shell boundary after writing 100 replay files, but replay-derived metrics showed average ante 4.11, ante 4+ on 69/100 runs, ante 5+ on 47/100 runs, 175 sell actions, and average played hands per blind improved from 2.40 to 2.14.
- Added first-pass survival-aware shop pressure: the bot estimates next-blind target score, scorer-based build capacity, and early-build joker pressure, then adjusts buy/reroll/replacement thresholds and interest penalties. Shop replay reasons now include pressure, target, and capacity.
- Tuned the pressure model after a bad first smoke that over-saved with incomplete early builds; the clean 20-seed validation averaged ante 3.75 with 13/20 runs reaching ante 4+, 8/20 reaching ante 5+, and 25 replacement sells.
- Added `balatro_ai.eval.replay_analyzer`, a compact replay summary tool for average max ante, ante reach rates, action counts, shop reason counts, pressure stats, sell actions, and played-hands-per-blind efficiency.
- Ran the tuned survival-pressure bot over 100 White Stake seeds with 4 workers: official benchmark win rate 1.0%, average ante 4.00, average final score 10,761.9, average final money 54.7, average runtime 50.25 sec/run. Replay analyzer covered 99 replay files with average max ante 4.04, ante 4+ on 67/99, ante 5+ on 44/99, 148 sell actions, and average played hands per blind 2.14.
- Made fast benchmarking a real low-overhead path: added summary-only replay rows, GUI/CLI summary replay selection, a `--fast-benchmark` CLI shortcut, and configurable start retries for intermittent bridge start failures.
- Started consistency analysis on deep losses and early busts: replay analyzer now reports early-failure and deep-loss sections, summary replays include final state/jokers, and future step replays include compact shop/pack/chosen-item details. Tested two first-shop tuning ideas on the same 20 seeds; both underperformed or failed to beat the old baseline, so no unproven bot behavior change was kept.
- Added structured shop decision audits to `basic_strategy_bot` action metadata: light/score-audit replays now record pressure, thresholds, visible options, chosen item, option values, replacement candidates, skips, and rerolls. Ran a fresh 100-seed, 4-worker light replay in `.data/shop-audit-light-100`: win rate 0.0%, average ante 3.60, ante 5+ on 25/100, ante 6+ on 9/100, 26 early failures, and 9 ante 6-7 losses. Saved the analyzer report to `.data/shop-audit-light-100-analysis.txt`.
- Improved blind-play tactics in `basic_strategy_bot`: play/discard actions now carry tactical reason metadata, on-pace hands are played unless a discard is expected to reduce hands needed, known-deck discard lookahead is used when available, and unknown-deck discard estimates now prioritize real strong draws instead of speculative redraws.
- Ran a 20-seed, 4-worker light replay after the tactical play update: benchmark average ante 3.65, analyzer average max ante 4.00, 0 wins, ante 5+ on 8/20, ante 6+ on 4/20, and average played hands per blind 2.16. The run exposed a late-shop reroll-cost legality gap, so `basic_strategy_bot` now keeps a larger bank before rerolling full late-game builds.
- Added build-role targeting to `basic_strategy_bot` shop decisions: the bot now tracks missing chips, +Mult, xMult, scaling, and economy layers, records that profile in shop audits, and boosts buys/packs/rerolls that fill missing late-game roles.
- Added archetype observability to replay analysis: played hand-type distributions, dominant hand type by run, shop preferred-hand signals, final preferred-hand signals, and missing build roles now appear in analyzer text/JSON output.
- Made blind discard selection archetype-aware: flush builds protect suited cores, straight builds protect connected ranks, and rank builds protect duplicate ranks. Replay state details now also persist the visible hand and hand levels for cleaner future archetype analysis.
- Added play-to-cycle behavior: when a play already meets the same scoring goal, `basic_strategy_bot` prefers legal plays that include low-value non-scoring cards so it can dig deeper through the deck without spending a discard, while avoiding cards the current archetype wants to keep.
- Ran a 100-seed, 4-worker White Stake summary benchmark after the archetype/cycle-play changes: win rate 3.0%, average ante 4.06, average final score 14,029.6, average final money 55.2, average runtime 75.13 sec/run, with 0 replay errors. Analyzer report saved to `.data/cycle-summary-100-analysis.txt`.
- Added late-shop spending gates: `basic_strategy_bot` now tracks rerolls/packs per shop, caps safe late role-hunt rerolls, and skips late packs unless pressure is high or the estimated scoring capacity improves after accounting for the money spent. A final 12-seed validation improved from the prior gated run's 1/12 wins and 4.83 average ante to 2/12 wins and 4.92 average ante, while keeping early ante <=2 deaths at 2/12.
- Added benchmark failed-seed retry handling: the runner now retires an unhealthy endpoint after a bridge/client error, finishes the main sweep on healthy endpoints, then retries only failed seeds and replaces their replay JSONL files. A 100-seed White Stake summary run completed with 0 final error summaries after retry: win rate 2.0%, average ante 4.19, ante 5+ on 46/100, and early ante <=2 losses down to 18/100.
- Added reserve-aware shop spending: `basic_strategy_bot` now treats money above the current interest cap as spendable, respects Seed Money/Money Tree raised caps, and protects larger reserves for money-scaling jokers such as Bull and Bootstraps. Shop audits now report interest cap, reserve, spendable money, and money-scaling status.
- Ran a 100-seed, 4-worker White Stake summary benchmark after the reserve-aware spending update using a fresh deterministic label: win rate 0.0%, average ante 3.77, ante 5+ on 38/100, ante 6+ on 15/100, and early ante <=2 losses at 28/100. Because this used a new seed label, treat it as directional rather than an exact same-seed A/B; the loud failure signal is still deep losses with high unspent money.
- Inspected early-death seed `1387607577` with score-audit replay and fixed two ante-1 mistakes: the bot now hunts for a one-hand first Small Blind clear with discards instead of accepting weak Two Pair pace, and early shops block rerolls when an affordable Buffoon Pack or strong visible joker/economy path exists. Rerunning the same seed improved from ante 1 death at 562/600 to ante 3 death at 2868/3000.
- Tightened the ante-3 follow-up from that seed: unknown-deck discard projections are now stricter when deciding to skip an on-pace hand, rank/flush joker archetypes now include larger compatible hands such as Wily Joker supporting 4/5 of a kind, and Joker Stencil is valued at zero when it would fill the last normal joker slot.
- Added rare-hand consistency logic to Phase 5 shop valuation: narrow rare-hand commitments such as The Family are discounted without visible duplicate/deck support, while active rare-hand builds value rank/suit deck manipulation, Arcana/Standard/Spectral packs, and matching playing cards more highly.
- Audited seed `1387607577` shop purchases and fixed two valuation mismatches: owned Joker Stencil now loses estimated value when a normal joker fills an empty slot, and compatible rare hands no longer become the primary plan just because multiple broad joker synergies overlap. The same seed now skips Wily Joker, keeps Stencil at X2, and reaches ante 4 Big Blind instead of dying on ante 3 Big Blind.
- Added starter-deck impossibility handling for rare hands: Five of a Kind, Flush House, and Flush Five now keep an extra manipulation requirement unless the run has evidence such as held supporting tarot cards, exact duplicate cards, or prior leveling in that hand.
- Added sell-side Joker Stencil accounting: when evaluating which owned joker to sell, the bot now credits Stencil for the empty slot created by that sale.
- Added a bounded panic-discard rule for blind play: when the best hand is far below the per-hand score pace and discards remain, the bot keeps digging instead of playing tiny hands and dying with unused discards.
- Tuned Phase 5 build planning away from weak Two Pair commitments: Two Pair support jokers no longer force Two Pair as the primary plan without a dedicated scaler such as Spare Trousers, Uranus is discounted without that support, pressure can justify spending below interest reserve for true scaling/xMult help, and unsupported trip/rare-hand jokers are penalized before they fill key joker slots.
- Added explicit blind-clear safety margins to shop and blind-play evaluation: shop pressure now tracks raw versus safety-adjusted target ratios, discounts fragile late-ante build capacity, records safety factors in audits, prefers visible buffer packs before full-slot rerolls, and blind play can spend discards when the best hand is below buffered scoring pace. Same-seed validation on `1387607577` improved the original ante-4 failure from `6855/7500` to `9448/10000`; it now spends into late Arcana/Celestial packs but still misses The Wheel by 552 points, so the next issue is pack-card/deck-fixing value rather than pure shop hoarding.
- Ran the primary 100-seed White Stake score-audit benchmark in `.data/primary-score-audit-100`: initial official benchmark win rate 1.0%, average ante 3.56, average final score 10,029.0, average final money 46.0, average runtime 64.45 sec/run. Fixed the full-joker-slot direct-buy legality bug, then fixed the remaining deterministic bridge wedge on seed `1094190033` in BalatroBot itself: the `buy` and `pack` endpoints no longer infer hand-loading requirements from the first revealed pack card, so legal Celestial packs that reveal special Spectral cards return normally. All 100 replay files now end in normal outcomes: 1 win, 99 losses, 0 final errors, average summary ante 3.90, average max ante 3.90, ante 4+ on 63/100, ante 5+ on 35/100, and ante 6+ on 15/100. Score audit now covers 2,398 played hands with 1,820 exact matches; the largest remaining scoring gaps are boss/conditional effects such as The Eye/The Mouth/Crimson Heart and high-value straight/The Order scenarios.
- Fixed opened-pack valuation for Black Hole: the bot now treats it as a premium all-hand planet upgrade, picks it over matching planets such as Jupiter, and does not reject it just because consumable slots are full. Rerunning seed `1094190033` confirms it now picks Black Hole from the Celestial pack and improves that run's final score from 4,494 to 6,656, though the run still dies on ante 4.
- Started Phase 7 foundation work: Photograph scoring now triggers during scored-card evaluation instead of normal joker resolution, with regression coverage for earlier XMult and Polychrome joker ordering.
- Added `balatro_ai.eval.compare`, a paired same-seed comparison CLI with win-rate flips, exact McNemar p-values, Wilcoxon signed-rank ante p-values, and bootstrap confidence intervals for score deltas.
- Added replay analyzer `--postmortem-summary` output that groups losses by failure ante, blind type, primary missing build role, and final money bucket.
- Added the first Phase 7 `balatro_ai.search` package surface with deterministic play/discard forward simulators. The simulators require callers to inject drawn cards and leave shop/pack/blind transitions as explicit not-yet-implemented steps until each transition type can be replay-validated.
- Added `balatro_ai.search.replay_diff`, a replay-diff harness for validating deterministic simulator transitions against JSONL replays. On `.data/primary-score-audit-100`, discard transitions currently match 1,293/1,298 exactly (99.6%) and play transitions match 2,237/2,399 exactly (93.2%). All 167 remaining mismatches are now labeled as known gaps from random/rounded effects, older Certificate snapshots, Obelisk reset history, or boss effects such as The Hook and The Pillar; the comparable deterministic subset is 3,530/3,530 exact.
- Extended future replay `state_detail` rows with raw hand-play metadata from `state.modifiers["hands"]`, so boss restrictions such as The Eye and The Mouth can be replay-diffed with less reconstruction loss in fresh score-audit runs.
- Closed the deterministic replay-diff scoring misses found in `.data/primary-score-audit-100`: Spare Trousers now triggers on full houses, The Flint rounds halved base chips/mult up like Balatro, The Tooth reduces money before Bull/Bootstraps score, scored-card jokers resolve during card scoring, disabled jokers no longer apply ability/edition effects, and The Hook visible-hand gap was isolated for later transition-oracle modeling.
- Updated score-audit uncertainty classification for isolated rows involving Certificate and The Eye. The recomputed miss explainer now reports 0 supported score misses on the 100-seed replay corpus; stored score-audit prediction totals still reflect the older evaluator that generated those replay rows.
- Preserved full joker metadata in future replay rows and added visible-counter support for Ice Cream, Popcorn, and Supernova. The replay-diff harness reconstructs their current values in the older 100-seed corpus from deterministic replay history, removing them from the known-gap bucket.
- Added more visible-counter handling for replay validation: Square Joker, Green Joker, Loyalty Card, and Ramen now consume logged current values where available, and the older corpus reconstructs those values from replay history. Square Joker scoring now applies its +4 chip gain before the current 4-card hand scores, matching Balatro's timing. Green Joker, Loyalty Card, and Card Sharp are no longer treated as replay-diff known gaps; the remaining Ramen rows are one-point fractional XMult rounding differences.
- Added a Phase 7 score-edge fixture suite under `tests/fixtures/score_edges` plus `balatro_ai.eval.score_edge_fixtures`. The suite runs exact evaluator fixtures, deterministic play/discard transition fixtures, and explicit known-gap markers for Obelisk, Certificate, The Hook, and The Pillar so future bridge-captured oracle rows have a clear home.
- Expanded the score-edge suite with rare cases that Basic Strategy may not naturally reach: Five of a Kind, Flush House, Flush Five, Four Fingers, Shortcut, Splash, stone-card hand typing, Blueprint/Brainstorm, disabled jokers, red-seal/glass/Photograph retriggers, Dusk retriggers, Mime/Baron/Steel held-card XMult, Raised Fist debuff handling, The Psychic, and The Mouth.
- Started Stage 3 deck modeling: future replay rows now preserve `known_deck`, `DeckModel` can build exact multisets from live state, sample without replacement, enumerate small draw sets, and remove seen cards for within-blind tracking. Added `balatro_ai.search.deck_model` replay draw validation; the old 100-seed corpus has 0 exact-model impossible draws and 45 inexact candidate misses caused by missing `known_deck`/deck changes in old logs.
- Expanded the offline score dataset to 50 deterministic fixture cases, adding card-edition/enhancement ordering, boss suit debuffs, permanent card chips, Seltzer/Hack retriggers, Seeing Double, Ancient Joker, The Idol, Triboulet, Baseball Card rarity fallback, and more visible-current counters such as Ice Cream, Popcorn, Green Joker, Driver's License, Obelisk, Lucky Cat, Joker Stencil, and Hit the Road. Added `balatro_ai.eval.score_dataset`, which validates fixtures and recomputed replay score-audit rows together; on `.data/primary-score-audit-100` it checks 1,782 supported replay rows with 0 misses.
- Launched multiple headless BalatroBot workers for a fresh Phase 7 score-audit pass. Two workers became healthy on ports 12347 and 12349; the other two hit a Lovely dump-directory lock. Captured new corpora under `.data/phase7-score-audit-smoke`, `.data/phase7-score-audit-legality-rerun`, and `.data/phase7-score-audit-fresh-10`.
- Fixed two issues found by the fresh live corpus: `basic_strategy_bot` now remembers when it filled the last normal joker slot in a shop so stale bridge snapshots cannot trigger an illegal follow-up joker buy, and The Arm now lowers the scored hand level before scoring. Cerulean Bell rows are now marked as a known uncertainty because the boss can force the actual scored cards away from the requested action cards.
- Revalidated the expanded score dataset across old and fresh replay data: 52 deterministic fixture cases, 111 replay files, 2,792 replay audit rows, 2,025 supported replay rows checked, 0 replay score misses, and 767 known-uncertain rows. The fresh replay-diff corpus is 407/407 exact on comparable deterministic transitions, with remaining mismatches labeled as Cerulean Bell, Misprint, Obelisk, or Ramen known gaps. This corpus also exposed the seed `310855324` live-run blocker fixed in the next item.
- Fixed the seed `310855324` live-run blocker. The root cause was the bridge reporting `ante=9` with `won=True` but `run_over=False`; the runner now treats any ante 9+ state as the standard White Stake win boundary and stops at ante 8. Empty booster-opened states are also hardened so they produce a `no_op` refresh instead of an invalid pack-skip RPC. Rerunning the seed now completes normally as a win at ante 8 with score 108,391 and money 109.
- Pulled a lightweight BalatroBench subset from Kaggle by extracting only raw `gamestates.jsonl`, `responses.jsonl`, `task.json`, and `stats.json` files instead of the full screenshot-heavy archive. Added `balatro_ai.eval.balatrobench_score_audit`, which infers executed play transitions directly from consecutive game states so invalid LLM tool calls cannot misalign the audit. The external corpus currently provides 2,148 inferred play records across 241 runs. After adding BalatroBench enhanced-card label parsing, visible `extra chips` parsing, Wild Card suit/debuff handling, live Popcorn/Ice Cream values, Smeared suit-joker effects, Midas/Hiker/Stone/Gold Seal ordering, leveled The Flint handling, disabled-joker hand-shape filtering, and Lucky Card uncertainty classification, the audit reports 2,132 exact matches overall and 2,010/2,010 exact matches on the supported deterministic subset, with 138 known-uncertain stochastic/high-context rows.
- Locked the 1000-seed White Stake Basic Strategy baseline in `baselines/basic_strategy_2026_05_02.json` and `baselines/basic_strategy_2026_05_02.jsonl`: 74/1000 wins, 7.4% win rate, 4.86 average ante, 22,046.0 average final score, 47.7 average final money, 0 replay errors. Postmortem output was saved to `.data/basic-strategy-baseline-1000-analysis.txt`; the summary-only baseline has no action transitions for replay-diff validation, so simulator validation continues to use the detailed score-audit corpora.
- Re-ran Phase 7 detailed validation after the baseline: score dataset still passes with 52 fixtures, 2,271 supported replay rows checked, and 0 replay misses. Detailed replay-diff across 115 old/fresh score-audit files is 9,200/9,515 exact on comparable deterministic transitions (96.7%), with the remaining large cluster mostly older cash-out money rows whose logged blind/ante labels can be stale around cash-out.
- Improved `state_value.clear_probability` rollouts with a conservative greedy discard step when the current best play is below clearing pace. A quick 40-blind calibration smoke moved most formerly pessimistic mid-bin states into high-confidence clear bins, and the full test suite passes.
- Added the Phase 7 shop sampler foundation. `shop_pools.json` is generated from the bundled Balatro Lua source and includes 150 jokers, 22 tarots, 12 planets, 18 spectrals, 32 vouchers, and 32 boosters. `ShopSampler` now samples source-derived shop-card slots, voucher slots, booster packs, joker rarity thresholds, editions, discounts, Overstock slot counts, duplicate filtering, visible reroll costs, and first-pass reroll EV via Basic Strategy's shop value proxy. Distribution/state tests pass, and the full test suite is green with 368 tests plus 10 subtests.
- Completed a fresh 10-seed detailed local-sim oracle corpus in `.data/phase7-local-sim-oracle-fresh-10` with 10/10 normal outcomes. The replay validator now uses the observed post-cashout money delta for validation-only cash-out transitions because the bridge's pre-cashout `round.dollars` surface can be stale at `ROUND_EVAL`. Fresh corpus exactness improved from 1,059/1,286 transitions (82.3%) to 1,162/1,286 (90.4%); discard, end-shop, choose-pack, reroll, rearrange, and sell are 100% exact, while the remaining largest clusters are play-hand scoring randomness/context, select-blind deck-size carry, and dynamic joker sell-value updates.
- Added first-pass Phase 7 shop beam search in `balatro_ai.search.shop_search`. `search_bot_v1` is now available through the bot registry and layers shop search on top of the existing discard/pack search while leaving `search_bot_v0` unchanged. The beam search simulates buy, sell, reroll, and end-shop transitions, treats open-pack and reroll as receding-horizon terminal decisions, values rerolls with `ShopSampler.reroll_ev`, and correctly allows negative jokers at full normal joker slots. Focused shop/search-bot tests pass, and the full suite is green with 373 tests plus 10 subtests.
- Tightened the first live `search_bot_v1` shop-search smoke issues. Discard search now prunes large action sets for bridge-speed safety, reroll EV evaluates sampled shops after paying the reroll cost so last-dollar rerolls cannot value unaffordable cards, sell paths must buy a visible joker immediately after the sale, same-shop bought jokers are protected from sale, and shop/forward-sim buy legality now rejects consumable cards when consumable slots are full. Replaying seed `1263323122` after these fixes completed normally and improved from an ante-3/error smoke to an ante-4 Small Blind loss at 4,278 score; the full test suite is green with 380 tests plus 10 subtests.
- Made Phase 7 shop search interest-aware. Basic Strategy's buy/pack heuristic already charged lost interest breakpoints through `_cost_penalty`, but the search layer now also applies opportunity cost to reroll EV and values shop leaf cash by projected next cash-out interest, raised interest caps, reserve shortfall, and above-reserve spendability. Same-seed smoke seed `1263323122` improved again to an ante-4 Big Blind loss at 6,358 score with 3 final money; the full suite is green with 382 tests plus 10 subtests. A 3-seed paired live compare is running under `.data/phase7-v1-interest-compare-3`.
- Added full-slot Buffoon-pack replacement behavior. `search_bot_v1` can now open a Buffoon pack at full normal joker slots, compare shown jokers against owned jokers, sell an owned joker during pack selection when the replacement is better, take negative jokers without selling, or skip the pack when all shown jokers are worse. The local BalatroBot `sell` endpoint was patched to allow `SMODS_BOOSTER_OPENED`, and live seed `1801824049` now executes the sell-then-pick sequence (Photograph -> Card Sharp) without a bridge error, ending normally at ante 4 instead of crashing. The full suite is green with 389 tests plus 10 subtests.
- Reworked shop replacement search away from narrow per-joker exceptions. The beam now rejects replacement sells unless a simulated sell-then-buy improves a dynamic replacement value built from sample build score, role coverage, money/interest value, and utility state; it also carries same-shop bought jokers as protected through the beam so it cannot plan to buy a joker and immediately sell it. Same-seed live smoke `1801824049` now reaches ante 6 under `.data/phase7-v1-dynamic-replacement-gate`, with the remaining gap looking like broader shop/value tuning rather than a hardcoded Photograph/Clever/Hallucination exception. The full suite is green with 392 tests plus 10 subtests.
- Added a benchmark wall-clock cap for pathological long seeds. `RunSeedOptions` and `BenchmarkOptions` now carry `run_timeout_seconds` separately from the JSON-RPC request timeout, the benchmark/compare CLIs expose `--run-timeout-seconds` with a 30-minute default, and the GUI has a matching "Run timeout" field. Timed-out seeds are marked with a retryable `error:RunTimeout` death reason so the existing failed-seed retry pass can recover them. The full suite is green with 394 tests plus 10 subtests.
- Added single-seed profiling with `--profile-path` and investigated the 7.4-hour baseline outlier seed `1122649218`. The stale bridge was alive but unhealthy, and an offline cProfile of the captured ante-4 state showed `basic_strategy_bot.choose_action` taking 157s because joker rearrangement tried all 720 six-joker permutations and rescored every legal play for each order. Joker rearrangement now stays exhaustive up to 5 jokers but uses bounded role/copy candidate orders at 6 jokers. The captured decision dropped to 15.5s, and a patched live rerun of the seed completed in 123.1s with profile output under `.data/phase7-speed-profile`; the profile showed 82% bot-decision time and 17.5% bridge/env stepping. The full suite is green with 397 tests plus 10 subtests.
- Patched the local BalatroBot Lua settings for bridge speed. `C:\Users\Wyatt\AppData\Roaming\Balatro\Mods\balatrobot\src\lua\settings.lua` now respects requested fast-mode `gamespeed`/`animation_fps` values and clamps queued `after`/`before`/`ease` event delays in headless fast mode or when `BALATROBOT_NO_ANIMATIONS=1`. A verification bridge log confirmed "No-animation event delay clamp enabled"; same-seed profiling on `1122649218` did not improve that specific run because bot decision time still dominates.
- Started the pure-Python run simulator in `balatro_ai.sim.local_runner`. The first pass has an env-like `LocalBalatroSimulator`, `run_local_seed`, and `balatro_ai.eval.local_benchmark` for bridge-free seed sweeps. It uses exact ordered deck state, shuffles between blinds, drives play/discard/shop/pack transitions through the existing forward sim, samples shops and booster contents from source-derived pools, supports deterministic boss blinds by default, and records benchmark-style `RunResult`s. Focused local-sim/shop/forward-sim tests pass, and a 3-seed `greedy_bot` local benchmark ran at about 0.09 sec/run. Current known limits: skip tags are not applied, stochastic boss effects are excluded from the default boss pool, many random joker procs are deterministic/pessimistic unless forward-sim injection is added, and bridge benchmarks remain the source of truth.
- Added `balatro_ai.sim.replay_validator`, a carried-state local-sim validator for detailed bridge replay JSONLs. It checks pre-state drift, simulates observed actions with replay-injected randomness, resyncs after divergences, and preserves hidden validation-only blind progress across cash-out rows. Source-truth checks against the local Balatro Lua dump corrected the shop/blind timing: cash-out opens shop on the visible Small Blind surface, while `round_resets.blind_states` determines the next selectable blind. The forward sim now sorts hands in bridge/game order after draws and honors Lua cash-out modifiers such as `no_extra_hand_money` and `no_blind_reward`. On `.data/phase7-score-audit-fresh-10`, local-sim replay validation is now 919/1,149 exact overall (80.0%), with discard 108/115, end_shop 144/146, choose_pack_card 53/53, buy/sell/rearrange 100%, and remaining misses concentrated in score evaluator gaps, cash-out money rows that need logged `current_round.dollars`, The Hook/Cerulean Bell randomness, stochastic joker removal, and old rows without exact deck snapshots. The full test suite is green with 410 tests plus 10 subtests.
- Added the cash-out oracle path for future detailed replays. The local BalatroBot bridge now exports source-truth round-eval data from `G.GAME.current_round` and `G.GAME.blind`, including total cash-out dollars, blind reward/score/type, interest settings, no-reward/no-hand-money flags, blind states, and `blind_on_deck`. Python state parsing preserves these fields in `state.modifiers["round"]`, replay logging writes them into `state_detail.round`, replay-diff reconstructs them, and `simulate_cash_out` uses the logged total dollars when present. This does not change old replay-corpus numbers because those rows lack the new oracle fields, but it should make the next fresh detailed run's cash-out money mismatches directly diagnosable. The full suite is green with 413 tests plus 10 subtests.
- Validated the local simulator against a fresh patched detailed replay with the cash-out oracle active. The local BalatroBot bridge now also exports owned voucher names from `G.GAME.used_vouchers`, and Python state parsing no longer treats `G.shop_vouchers` as owned vouchers. The carried validator now injects observed round/shop/pack surface payloads, applies shop-bought passive hand/discard deltas before the next blind without double-counting, initializes visible dynamic joker counters such as Ice Cream/Popcorn/Ramen when acquired, and classifies Crimson Heart's source-confirmed random joker debuff as a known stochastic boss gap. On `.data/phase7-local-sim-oracle-smoke-voucherfix`, validation is 182/184 exact overall (98.9%) with 0 comparable divergences; buy, cash_out, discard, end_shop, reroll, select_blind, sell, open_pack, choose_pack_card, skip_blind, and rearrange are all 100% exact. The only remaining rows are known Crimson Heart randomness. The full suite is green with 417 tests plus 10 subtests.
- Continued local-sim validation on the 10-seed detailed oracle corpus. Source-truth Seltzer handling now parses the visible "next N hands" countdown and destroys the joker after the final played hand, and the carried validator now starts each transition from the observed hand/deck surface so known stochastic drift from prior rows cannot pollute later deterministic checks. The one impossible summary row with `won=true` below the Ante 8 required score is now classified as a replay data gap rather than a simulator miss. On `.data/phase7-local-sim-oracle-fresh-10`, validation is now 1,285/1,286 exact overall (99.9%) with 0 comparable divergences; buy, cash_out, choose_pack_card, discard, end_shop, open_pack, rearrange, reroll, select_blind, and sell are all 100% exact, and play-hand transitions are 359/360 exact. The only remaining mismatch is the inconsistent run summary. Pack-search leaf evaluation now keeps temporary pack hands for heuristic valuation only, while replay/real forward-sim transitions still return those hands to the deck. The full suite is green with 437 tests plus 10 subtests.
- Source-checked Overstock against Balatro's `change_shop_size` Lua path and updated the local shop simulation to match it: buying Overstock or Overstock Plus now preserves any remaining visible shop cards and fills every empty shop slot up to the new shop max, not just the single newly added slot. `ShopSampler.fill_shop_to_slot_count` powers this behavior in both the pure-Python local runner and shop beam search, while replay validation can still inject the exact observed post-buy shop surface. Focused shop/local-sim tests pass and the fresh oracle replay validator remains at 1,260/1,286 exact with 0 comparable divergences.
- Source-checked The Hook against Balatro's boss-blind Lua and modeled it in the local simulator. Forward sim now accepts injected Hook-discarded held cards, replay validation reconstructs those cards from the observed post-hand surface when possible, and the local runner samples the discard before calculating replacement draws. The Hook is no longer a transition-level known gap; its last apparent miss is now correctly classified as Misprint randomness.
- Source-checked Misprint against Balatro's joker Lua and modeled its 0-23 random Mult. The local runner now samples Misprint play outcomes, and replay validation infers the observed Misprint Mult from the post-score before comparing the transition. This lifts the fresh oracle replay validator from 1,260/1,286 exact (98.0%) to 1,282/1,286 exact (99.7%) with 0 comparable divergences.
- Source-checked Mail-In Rebate and Cerulean Bell against the Balatro Lua. Mail-In Rebate rank parsing now reads the discarded-card rank from the visible effect text instead of mistaking the `$5` payout for rank five, and Basic Strategy's discard valuation uses the same parser. Replay validation now handles Cerulean Bell by trying the source-truth one-card forced selection that Balatro applies on draw-to-hand, and it mirrors the final Ante 8 win flag when the simulated final boss clear reaches `ROUND_EVAL` with enough score. Fresh oracle validation is now 1,285/1,286 exact (99.9%).
- Fixed the standard Ante 8 win-boundary normalization. The runner and raw bridge state parser still treat nonterminal `ante=9` cleanup states as White Stake wins, but they no longer convert an already-terminal `ante=9` game-over loss into `won=true`; this explains the single inconsistent old replay summary and prevents future summary rows from repeating it. The full suite is green with 439 tests plus 10 subtests.
- Completed the fresh 20-seed, 8-worker, 64x detailed oracle run in `.data/phase7-local-sim-oracle-fresh-20-20260502-64x`, then reran the one stale `_normalize_rank` error seed under the fixed code so the corpus now has 20 normal summaries, 2 wins, and 0 errors. Investigating the slow tail showed that completed/old timed-out 64x bridge workers kept spinning at the menu or in stale runs and starved the active seed; the benchmark runner now parks finished parallel endpoints with a `menu` RPC before returning them to the idle pool, with a CLI escape hatch via `--no-park-finished-endpoints`. Forward sim now applies end-of-round Egg/Popcorn maintenance when the final played hand ends the blind, including losses, removing the fresh corpus joker-state mismatches. Current local-sim replay validation on the 20-seed corpus is 2,076/2,089 exact (99.4%) with 6 comparable divergences left, all in scoring/required-score/win-flag buckets already called out by the validator.
- Closed the remaining comparable fresh-20 local-sim mismatches. Source-checking the bundled Balatro Lua confirmed The Needle has `mult = 1`, so local blind surfaces now use the small-blind score for Needle bosses. Hook-forced held-card discards now run the same discard joker context before scoring, disabled jokers no longer advance play/discard counters, explicit simulator metadata takes precedence over stale visible joker text, and replay validation can infer blind-clearing Hook discard pairs from the observed score. Score flooring now tolerates tiny decimal XMult machine noise without masking real Ramen fractional floors. The fresh 20-seed oracle corpus validates at 2,087/2,089 exact (99.9%) with 0 comparable divergences; the only remaining rows are known Crimson Heart stochastic/hidden state and one stale win-summary data gap. The full suite is green with 448 tests.
- Fixed Crimson Heart as an explicit simulator outcome instead of a broad stochastic boss gap. Forward sim and replay validation now carry the next disabled Joker index after a played hand, and the local runner samples Crimson's source-truth behavior: clear old debuffs, exclude the currently debuffed Joker when there are at least two choices, then debuff the next Joker for the following hand. Future replay rows now preserve card metadata so hidden scoring-card state such as permanent chips can be audited. The fresh 20-seed oracle corpus remains at 2,087/2,089 exact with 0 comparable divergences; the remaining Crimson row is now classified as an older-log hidden card metadata gap, not a disabled-Joker mechanics gap. The full suite is green with 450 tests.
- Filled in the remaining high-impact local-sim joker mechanics against the bundled Balatro Lua source: random play procs for Bloodstone, Business Card, Reserved Parking, Space Joker, Lucky Card/Lucky Cat, 8 Ball, glass shatter, Gros Michel, and Cavendish; generated consumables from Sixth Sense, Superposition, Seance as Spectral, Vagabond, Hallucination, and Perkeo; round targets for Ancient Joker, The Idol, Castle, and Mail-In Rebate; blind-select growth/destruction for Ceremonial Dagger and Madness; Invisible Joker sell duplication; Diet Cola's Double Tag sell path; Caino/Glass Joker destruction growth; Hologram, Constellation, Satellite usage state; and deck-derived values for Joker Stencil, Stone Joker, Steel Joker, Driver's License, and Erosion. The full suite is green with 455 tests, and the fresh 20-seed oracle corpus still validates at 2,087/2,089 exact with 0 comparable divergences.
- Added a bridge-backed joker scenario oracle. The local BalatroBot mod now has a dev `scenario` endpoint that can replace the visible hand and owned jokers before normal bridge actions run, and `balatro_ai.sim.bridge_joker_smoke` generates one controlled oracle replay scenario for every joker in `shop_pools.json`. The manifest dry-run covers all 150 base jokers; live bridge workers must be restarted before using the new endpoint.
- Expanded the bridge-backed joker smoke oracle to 185 scenarios and validated the full suite against a live 64x bridge with 185/185 exact simulator transitions. The scenario endpoint can now force active boss blinds and probability multipliers, keeps forced boss blind metadata/status aligned with the live game, and refreshes boss debuffs after injected cards are added. The smoke suite now includes forced stochastic procs, boss-blind edge cases, shop-event jokers, and rare interaction stacks; replay-diff also uses deck deltas to resolve ambiguous Hook discards when an injected held card is discarded and an identical card is drawn back.
- Added source-truth skip tag handling to the pure-Python local simulator using the Balatro Lua `tag.lua`/`game.lua` timing buckets. Blind-select surfaces now expose Small/Big skip tags, skip actions add pending tags and honor Double Tag copying, immediate tags apply Economy/Skip/Handy/Garbage/Top-up/Orbital effects, Boss Tag rerolls the upcoming boss, pack tags open free tag packs and return to blind select, Juggle applies a temporary round hand-size boost, Investment pays on boss cash-out, and shop tags handle D6, Voucher, Coupon, Uncommon/Rare, and edition joker effects. `ShopSampler` now has targeted joker-rarity and booster-kind samplers for these tag effects. The full suite is green with 493 tests plus 10 subtests, and a one-seed local Basic Strategy smoke completed normally.
- Added the remaining source voucher effects and expanded boss pools in the pure-Python simulator. Voucher purchases now apply source-derived shop rates, edition/play-card rates, reroll-cost reductions, interest caps, joker/consumable slots, hand-size changes, Hieroglyph/Petroglyph ante rollbacks, Observatory scoring, Omen Globe spectral Arcana rolls, Telescope first Celestial planet targeting, and Director's Cut/Retcon boss reroll legality. Local boss selection now uses the full source normal/showdown boss pools with source min antes, score multipliers, rewards, and least-used selection. The full suite is green with 502 tests plus 24 subtests.
- Filled in the remaining source boss behavior in the local simulator. The Manacle now applies/restores its temporary hand-size reduction, The House/The Fish/The Mark apply their face-down draw rules, The Serpent draws exactly three cards after play/discard, The Pillar debuffs cards played earlier in the ante, The Plant and Verdant Leaf apply card debuffs through visible card state, Verdant Leaf can be disabled by selling a joker during the blind, The Ox resets money when the run plays its most-played hand, Cerulean Bell marks a forced selected card and legal actions must include it, and Amber Acorn shuffles/flips jokers on boss start. The full suite is green with 510 tests plus 24 subtests, and a one-seed local Basic Strategy smoke completed normally.

### 2026-05-24 to 2026-05-25 — Pivot to offline solver path

- **Architectural pivot (2026-05-24).** Stopped trying to make `basic_strategy_bot` win and started building toward an offline expert solver for Phase 8 training data. Rationale and full plan in `PHASE7_OFFLINE_SOLVER_PLAN.md`. The leaf-tuning loop had plateaued at ~5-7% white-stake winrate across many search variants; the gap to the Phase 8 40-50% gate isn't closable by incremental tuning.
- Added `balatro_ai.eval.sim_divergence_audit`, a new diagnostic that feeds full prior state into `simulate_*` and compares each post-state field against actual BalatroBench post-states. Initial run revealed forward_sim is 99.9% exact on play/discard/sell/reroll/end_shop (5070/5074 transitions across 241 runs), not the much-lower number earlier action-replay audits implied — those measured audit reconstruction limits, not sim bugs.
- Found and fixed two real sim bugs from the divergence audit: The Arm boss blind didn't permanently decrement stored hand levels (24 cases), and To Do List `$4` payout was completely missing (10 cases; mid-fix discovery is that it pays on EVERY matching play, not just first hand of round). Both fixed at `forward_sim._hand_levels_after_play` and `hand_evaluator._to_do_list_target` + integration.
- Built seed-faithful RNG infrastructure in `src/balatro_ai/rng/`: `pseudohash.py` (Balatro's pseudohash + pseudoseed_step with the `string.format("%.13f", ...)` round-trip), `balatro_rng.py` (per-key `BalatroRNG` class with `mix_hashed_seed=True` default), `luajit_prng.py` (LuaJIT TW223 bit-exact port with the `d * pi + e` seeding transform), `deck.py` (verified pre-shuffle order `C,D,H,S × 2..9,A,J,K,Q,T`), and `pools.py` (Tarot/Planet/Joker pools extracted from `game.lua`).
- Added capture/validate harness: `rng/capture.py` and `rng/capture_shop.py` produce ground-truth fixtures via the bridge; `rng/validate.py` runs grid search over candidate algorithms. Captured fixtures live under `.data/rng-validation/`.
- Critical bug found during validation: `pseudohash` was using the iteration counter as the position multiplier, but Lua's `for i = #str, 1, -1` keeps `i` as the visited character's original 1-indexed position. Single-char strings (which my unit test used) happened to match accidentally. Fixed in `pseudohash.py`.
- **Deck shuffle SOLVED (2026-05-25):** algorithm is `luajit_after_pseudoseed` (one `pseudoseed('shuffle')` call seeds LuaJIT TW223, then Fisher-Yates with `math.random(i)`), matching all 4 captured fixtures on the full 52-card deck.
- **Ante-1 shop pool SOLVED (2026-05-25):** `predict_first_shop(seed)` in `src/balatro_ai/rng/shop.py` predicts category + specific item for all 4 seeds × 2 slots (8/8 exact). Required understanding Steamodded's type ordering `[Joker, playing_card, Tarot, Planet, Spectral]` and the rarity-then-pool subsequent pseudoseed advances.
- Built a Lovely probe mod at `C:\Users\Wyatt\AppData\Roaming\Balatro\Mods\rngprobe\` that hooks `create_card_for_shop` to log actual `pseudoseed` values to `%APPDATA%\Balatro\rngprobe.log`. Was instrumental in confirming Python predictions match the live game digit-for-digit. Reusable for future RNG-path validation.
- Finished first-pass seed-faithful RNG surface coverage in `balatro_ai.rng.surfaces`: initial boss/voucher/Small+Big tags, shop cards, booster slots, booster contents for Buffoon/Celestial/Arcana/Standard/Spectral, joker edition/sticker polls, standard-pack card fronts/seals, and per-card spectral helpers. `tests/test_rng_surfaces.py` validates the four canonical first-shop fixtures across boss/tags/voucher/shop cards/boosters.
- Added opened-pack RNG fixture tooling: `balatro_ai.rng.capture_surfaces` opens visible first-shop packs or forces one normal pack per kind through the dev scenario endpoint, and `balatro_ai.rng.validate_surfaces` compares captured `pack_seed_*` fixtures to `predict_pack_contents`. `tests/test_rng_pack_surfaces.py` covers the normalizers and skips safely until live pack fixtures are captured.
- Added no-purchase multi-shop RNG fixture tooling: `balatro_ai.rng.capture_shop_sequence` uses the dev scenario endpoint to clear blinds without buying, and `balatro_ai.rng.validate_shop_sequence` validates carried RNG state across repeated shops. Captured all four canonical seeds for six shops each on White and Gold Stakes; 48/48 shop-card surfaces and booster slots match through the first ante-3 shop. This exposed and fixed source `enhancement_gate` filtering for Steel Joker, Stone Joker, Lucky Cat, Golden Ticket, and Glass Joker, plus joker compatibility filtering for eternal/perishable sticker rolls.
- Added Spectral helper fixture tooling: `balatro_ai.rng.capture_spectral_helpers` forces Familiar, Grim, and Incantation in a controlled hand, and `balatro_ai.rng.validate_spectral_helpers` checks the created enhanced cards against `predict_spectral_created_cards`. The seed `AAAAAAA` fixture pass caught and fixed two source details: Familiar creates 3 cards, and Familiar/Grim/Incantation exclude Stone from the enhancement pool.
- Extended voucher-influenced RNG validation. The dev scenario endpoint can now set owned vouchers and played-hand counters, so live fixtures validate Omen Globe Arcana spectral rolls, Telescope first-Celestial planet forcing, Glow Up Standard-pack edition rates, and Magic Trick/Illusion shop-rate paths. `validate_surfaces --all` now passes 24/24 opened-pack fixtures and `validate_shop_sequence --all` passes 51/51 shops. One remaining edge was exposed and left explicit: Illusion shop playing-card generation advances global `math.random`, which affects the first Buffoon pack path.
- Updated project plan documents: created `PHASE7_OFFLINE_SOLVER_PLAN.md` with the new architecture, work-done summary, and roadmap. Updated `README.md` Current Status and Next Target sections.

### 2026-05-26 to 2026-05-27 — Rust core port (Phases 1-4a)

- **Phase 1 (state representation):** scaffolded `botlatro-core/` crate with PyO3 0.22 + ABI3 for Python 3.11+. Ported `Card`, `Joker`, and `GameStateNative` with `from_python`/`to_python` round-trip. Maturin + `pip install ./botlatro-core` builds and installs the native extension on Windows.
- **Phase 2 (hand evaluation):** ported `identify_hand_type`, `scoring_indices`, `card_chip_value`, and a composed `evaluate_simple` end-to-end. 75x speedup vs Python on standalone 5-card hands. Wired into `state_value._score_action` as the fast path (Phase 2 wire-in).
- **Phase 2d batches 1-20:** ported ~80 jokers across 20 incremental batches. Coverage includes simple ability/per-card jokers, scaling counters (Green Joker, Ride the Bus, Square Joker, etc. via JokerMetadata), held-card pass (Steel, Mime, Baron, Shoot the Moon, Raised Fist), card editions (Foil/Holographic/Polychrome), joker editions, ctx-aware (Card Sharp, Supernova, Acrobat), suit-debuff bosses, target-suit/rank (Ancient Joker, The Idol), multi-joker interactions (Swashbuckler, Baseball Card), identification modifiers (Pareidolia, Smeared, Four Fingers, Shortcut), Wee Joker (retrigger-aware), Stone cards, Wild cards, and Blueprint/Brainstorm copy-effect resolution. 117 parity tests + 202 Python-side Rust tests + 69 cargo tests all green.
- **Phase 3 (forward simulation):** ported `_draw_from_deck`, `_jokers_after_play` (~12 scaling jokers), `_jokers_after_discard` (5 jokers), `next_phase` decision, `_held_end_of_round_money_delta`, `_discard_money_delta`, `_hand_levels_after_play`. Top-level `simulate_play_simple` orchestrates the simple-case fast path (no Vampire/Midas Mask/Hiker/DNA/etc.), wired into Python's `simulate_play` with a 25.7% hit rate on AAAAAAA. Per-helper wire-ins didn't beat FFI conversion overhead; the architectural finding is that Phase 3 helpers are **scaffolding for Phase 4** (native search) rather than per-call wins from Python.
- **Phase 4 (4a-g + 4d.1 complete):** added `botlatro-core/src/search/scorer.rs::score_play_actions_batch` for batched action scoring — 16.58x faster per-state vs Python's per-action loop. Wired into legacy beam's `_cheap_beam_play_scores` (hand_search.py), the rollout's hot loops in state_value.py, shop-search build valuation, score_projection, and play_scoring. Phase 4c added `best_play_action_native` that combines enumerate+score+argmax into one FFI call. Phase 4f added an identity-keyed joker-data extraction cache (with tuple-ref guard against id() collisions). Phase 4g centralized the shared scaffolding in `search/rust_bridge.py`. **Phase 4d.1 took the architectural step: native `clear_probability_native` runs the entire greedy-rollout loop in Rust** — each call internally runs N rollouts × ~5 plays each with no FFI between steps, using an internal xoshiro256** RNG for deterministic draws (Python's random.Random parity is broken inside rollouts but clear_probability is an estimator — bot decisions remain quality-equivalent and trajectory parity is preserved). Restricted to `_RUST_BLIND_SAFE` blinds; boss blinds with scoring effects (Flint, Arm, Tooth, Psychic, Eye, Mouth, Plant) fall back to Python. **Trajectory on AAAAAAA: 49.4s vs 236s baseline → 79% speedup (4.78×)**, parity preserved (130 steps, RUN_OVER reason identical across all runs). The 3× acceptance gate from the original Phase 4 spec is MET 1.6× over. Path here: widened the rollout's blind-safe set (4d.1+ Manacle/Wheel/Psychic), wired `_evaluate_play_action` to Rust with a partial-HandEvaluation synthesis (4h), widened the bridge's evaluator safe set after auditing which bosses affect evaluator scoring vs forward_sim (4i: added Eye/Mouth/Manacle/Wheel and later Hook since these are forward_sim-only or applied by callers via _boss_adjusted_score; kept Psychic bailed since it DOES affect evaluator scoring), and **activated `decision_cache_scope` in `SolverPolicy.choose_action` (4j)** — found that solver path was running every `_identity_cached_value` factory because no cache scope was active, causing `_freeze_for_cache` to run 11M times per trajectory. One-line fix dropped trajectory 73s → 49.4s. **Phase 4d.2 (native beam recursion) attempted twice, both broke parity — `beam.rs` stays as scaffolding for a future side-by-side instrumentation port toward chess-engine-style deep search.**

### 2026-05-28 to 2026-05-29 — Live-bot winrate side-quest (early-build aggression + opt-in Rust play-search)

- Took the de-prioritized live-bot path (Next Steps #7) for a focused winrate push on `basic_strategy_bot`, White stake. On the small 100-seed set (`0000001`..`0000100`) the current bot baselines at ~14% (par harness) / 15% (config harness); these seeds run hotter than the 1000-seed corpus (7.4%), so treat the absolute numbers as set-relative.
- **Play-quality audit** (`scripts/play_quality_audit.py`): play is near-optimal — of 27 last-hand losses only 1 was recoverable by a different play. Build power, not play, is the bottleneck, so winrate effort belongs in shop/build decisions.
- **Phase 8 learned value function (probe) — NEGATIVE result, but diagnostically valuable.** Built a pure-numpy pipeline: `ml/features.py` (26 state features from the bot's own build signals), `scripts/phase8_gen_data.py` (9,438 shop-entry states / 400 seeds, labeled by run outcome), `scripts/phase8_train.py` (logreg, val AUC ~0.72, seed-level split, guidance-mode feature masking), `ml/value_model.py`, and a shop-buy hook `bots/basic_strategy/value_guidance.py`. Adding the model's 1-step ΔV as a shop-buy bonus REGRESSED winrate (9-10/100) at every scale. Root cause: the heuristic `_joker_card_value` is simulation-backed and joker-identity-aware — a superset of what a linear model over aggregate role scores can know — so the model is redundant and only adds noise. Hook left env-gated OFF (`BALATRO_VALUE_MODEL`). The real payoff was the model's calibration-by-ante: it showed the bot is OVER-OPTIMISTIC at antes 1-2 (predicted 0.23-0.48 win-prob vs actual 0.15-0.17), i.e. it under-builds early.
- **WIN — `shop_target_safety_base` 1.15 → 1.30** (new default in `bots/config.py`). Acting on the over-optimism finding, made the bot demand more score headroom before settling. Causal A/B via the new `scripts/winrate_bench_config.py` (BotConfig override on identical seeds): in-sample (seeds 1-100) 15→17, out-of-sample (101-200) 10→17; combined 25/200 (12.5%) → 34/200 (17.0%), stable 17/17 on both halves, never below control, +6 runs reaching ante 8. Canonical par harness 14→16. Benefit is early-only: pushing to 1.40, raising `shop_safety_cap` to extend it past ante 2, a `shop_value_tolerance`+`hand_pace_safety_base` combo, and `joker_sample_coefficient`=0.14 all regressed or were within noise. NOTE: measured on the small seed sets; a fresh 1000-seed confirmation is still pending.
- **Opt-in Rust play-search fast path.** `best_play_from_hand` (the bot's hottest loop, ~333K Python hand-evals/game) can route per-subset scoring through `balatro_core.score_play_actions_batch` via `search/rust_bridge.rust_best_play_scores`, building the full Python `HandEvaluation` only for the winner: **2.1× faster (26.6→11.0 s/game)**. Gated `BALATRO_RUST_BESTPLAY` (DEFAULT OFF) — the Rust simple-eval diverges from the Python evaluator on stateful jokers (Ride the Bus / Bull / Banner / Blue Joker / The Family; Rust models the play-content reset the Python projection ignores), shifting ~1.5% of decisions and moving the 100-seed winrate 14→11. Canonical bot stays bit-for-bit pure-Python; parity tooling at `scripts/bestplay_parity_check.py` (`BALATRO_BESTPLAY_PARITY=1`).
- Tooling added this stretch: `scripts/winrate_bench_par.py` (parallel winrate), `winrate_bench_config.py` (causal knob A/Bs + seed offset), `play_quality_audit.py`, `bestplay_parity_check.py`, `phase8_gen_data.py`, `phase8_train.py`. Updated one brittle pinned-value test (`test_joker_stencil_is_worthless_when_it_fills_last_slot`); 229 bot + 247 play-search/rules tests green.

### 2026-05-30 to 2026-05-31 — Data-gen speed + offline-solver winrate (churn + Buffoon fixes)

- **Data-gen speed pass.** Profiled the offline-solver data-gen path with `scripts/phase_timing.py` (process_time accumulators, parallel) — cProfile misleads here (over-weights Python many-small-call functions, blind to time spent in Rust). Cost is diffuse across the beam machinery (expand+enum ~42%, Python rollout-bail ~28%, headroom ~13%); no single hot spot. **WIN: `play_width` 2→1 in the dataset CLI + worker — 1.85× faster data-gen, quality-neutral** (ante 3.42 vs 3.44 over 96 seeds; depth=3 lookahead preserved). Native root-beam, rollout un-bails, and a canonical best-play-finder were each built and measured NEGATIVE for speed (rollout already in Rust; FFI/alloc overhead) and reverted or kept as scaffolding. Full findings in memory `project_datagen_speed.md`.
- **CRITICAL: the data-gen `SolverPolicy` winrate was ~1%** (vs `basic_strategy_bot` ~19-23%) — most generated trajectories were short losing games, bad training data. Started a winrate push (mandate: rules-correct + fast + high winrate; rewrite anything).
- **Play value-function bug FIXED (commit ec9d0b7).** `state_value._planning_value_uncached` valued a CLEARED blind (~1.03) BELOW an almost-cleared state still holding a strong hand (~1.39, inflated headroom score-component). The solver deferred its good hands to keep them "available" and ran out of hands. Fix: cleared blind returns `1.75 + min(0.25, headroom*0.25)` so clearing strictly dominates. Ante 3.46→4.26, ante-1 collapse cured. Found via `scripts/beam_decision_trace.py` (per-candidate value dump).
- **Shop joker-CHURN bug FIXED (commit f2944d8) — ~0% → 8% winrate.** Found via `scripts/shop_decision_trace.py` (per-action leaf terms + a BEAM candidate-path dump): the depth-2 shop beam SELLS a good joker, then RE-BUYS into the freed slot, netting positive search-score despite a strictly WORSE build. Root cause: `shop_action_search_value`'s BUY value (`_basic_shop_action_value`) is state-relative ("how badly do I need a joker now"), so selling first inflates the rebuy's heuristic value — and SELL returned `max(0,...) ≥ 0`, never charging the destroyed joker. Fix: charge the sale `sell_value − owned_value·0.45` (no floor; env `BALATRO_SELL_OWNED_COEFF`). Validated: 96 numeric seeds 0/96 → 8/96 wins, ante 4.25→4.74; 80 PAIRED seeds 2→6 wins (`scripts/shop_paired_ab.py`). Same bug CLASS as the play-value fix — a systematic search mis-valuation, not tuning.
- **First-shop BUFFOON-pack bug FIXED (winrate validation in progress).** Real Balatro guarantees a Buffoon pack (free starting joker) in the very first shop; the data-gen sim never produced one. Two compounding causes: (1) the data-gen harness runs the sim with the shop SAMPLER, not the seed-faithful path — the seed-faithful RNG only activates when a `balatro_seed` string is passed, which the harness omits (`_balatro_rng is None`); (2) the sampler's `sample_boosters` had an INVERTED guard (`first_shop_buffoon is False`) that was never reached (the flag is always True). So every data-gen game was denied its guaranteed early joker → chronic under-building. Fix: `sample_boosters(..., first_shop=...)` forces a Buffoon at slot 0 when it's the actual first shop (`_shop_index == 0`, threaded from `_cash_out`); env `BALATRO_FORCE_FIRST_BUFFOON`.
- **Structural negatives (save future effort):** leveling-based shop-leaf terms are INERT — they're constant across nearly every shop action, so they never move the search argmax (a leveling-concentration term left 38/40 paired games byte-identical). Joker-archetype coherence barely engages (only ~35% of bought jokers are in any archetype; Full House — the #1 leveled hand — is in none). Shop search depth 2→3 is WORSE (5→0 wins; deeper search amplifies the leaf's systematic errors). To steer BUILDS, change how JOKER PURCHASES are valued, not leveling or depth.
- **Ante-1 death audit** (`scripts/solver_ante1_audit.py`): 14/96 games die at ante 1, 12 of them at the ANTE-1 BOSS (600 target) with only ~1 joker — under-building. Many are near-misses (596/600, 492/600, 430/450). This drove the Buffoon-pack and early-build-aggression work.
- Tooling added: `shop_decision_trace.py`, `shop_paired_ab.py` (paired same-seed A/B — cancels the large seed-to-seed variance; the right design for small winrate effects; use ≥80 seeds for win counts), `shop_depth_ab.py`, `solver_build_audit.py`, `solver_death_analysis.py`, `solver_ante1_audit.py`, `buffoon_ab.py`, `datagen_speed.py`, `phase_timing.py`. Solver + shop tests green (256 + 56).

## In Progress

- **Offline-solver winrate push (active).** `SolverPolicy` data-gen winrate ~1% → ~8% after the play-value (ec9d0b7) and shop-churn (f2944d8) fixes; first-shop Buffoon fix landed and validating. The remaining gap to ~23% is build SCALING: deaths are blowouts (median ~50% of target) mostly to NON-boss Small/Big blinds at antes 4-5. The lever is leaf-value QUALITY (more systematic mis-valuations, found via `shop_decision_trace.py`), NOT more search depth. See memory `project_datagen_speed.md`.
- **Phase 4b-d (native search):** the Rust-batched scorer is the first incremental win. Remaining: `enumerate_legal_plays_native`, native single-subtree beam rollout, and full `solver_beam_play_action_native`. The acceptance gate is `≥3×` full-trajectory speedup; per-call wire-ins won't reach that — the search itself needs to live in Rust so state stays native and conversion happens once per search.
- **Validating expanded seed-faithful RNG coverage** against fresh bridge fixtures: the major pack/shop voucher paths now pass; remaining RNG work is the narrower Illusion playing-card/global-PRNG carry and optional Overstock slot-count fixtures. See section 4.1 of `PHASE7_OFFLINE_SOLVER_PLAN.md`.
- **Keeping `forward_sim` validated.** The divergence audit is the regression gate for any new sim work. Currently 99.9% exact; the two deferred minor bugs (Drunkard mid-shop sell, Credit Card negative-money reroll) are ~30 minutes each but not blocking the solver path.

### 2026-05-31 — Phase 8 neural architecture: plan + Step 0.1 (state encoder)

- Committed the end-goal architecture: a neural-guided, self-improving planner
  (AlphaZero/NNUE-style) on top of the exact forward model. Reframed the problem
  as a **deterministic single-agent puzzle** (seed-faithful RNG → perfect
  information) → best-first/A*/PUCT with a learned heuristic, not minimax. Key
  insight: a cheap learned eval replacing the rollout leaf is **faster and
  stronger at once** (the rollout is ~28% of data-gen runtime AND the source of
  the ante-8 ceiling). Full plan in `PHASE8_NEURAL_PLAN.md`; the solver and
  `basic_strategy_bot` are now bootstraps, not the destination.
- Data-gen speed audit (memory `project_datagen_speed.md`): box is 8 physical /
  16 logical cores; solver data-gen ~74–83s CPU/run at ante ~5.5 (the old 35.9s
  headline was at ante 3.44 — per-run rose because survival deepened, not a
  regression). ~345–420 seeds/hr; renting ~32 cores makes a 10k–50k dataset a
  ~$3–50 job; the project ports cleanly to Linux (pure-Python pkg + one
  dependency-free PyO3 crate; no GPU/game/bridge for data-gen), gated by a
  cross-platform determinism check.
- **Step 0.1 DONE — learnable state encoder** (`src/balatro_ai/ml/encoding.py`,
  `ENCODING_VERSION=1`). Structured, versioned, UNK-safe encoding of the RAW
  inputs the heuristic (and `observations.py` / the deprecated `ml/features.py`
  probe) throw away: joker identities (by display name) + editions + counters +
  flags, per-card rank/suit/enhancement/edition/seal, shop-item identities, hand
  levels, deck composition, boss id, and global scalars. Vocab from canonical
  data (joker=152 incl. legendaries, consumable=54, voucher=34, item-key=236,
  boss=30). Dependency-free (stdlib) so it tests without torch. 13 tests in
  `tests/test_ml_encoding.py` cover identity capture, index-bounds safety,
  UNK-safety, padding/empty states, hand-level mapping, and determinism — green.
  (Hard-won lesson: the real `GameState` API differs from a naive guess —
  `GamePhase` not `Phase`, jokers carry `name` not a key, no `ShopItem` (shop
  items are dicts in `modifiers["shop_cards"]`), `hand_levels` is a direct field.
  Verify state field shapes before extending the encoder.)
- **Step 0.2 DONE — training-data pipeline** (`src/balatro_ai/ml/dataset.py`).
  `capture_run` records a thin but **replay-complete** action log
  (`Action.to_json()` per step) + outcome; `replay_states` re-simulates that log
  on a fresh deterministic sim to reconstruct per-step states (no policy/search);
  `examples_from_capture` encodes them (Step 0.1) and attaches Monte-Carlo outcome
  value targets (`won`/`final_ante`/`final_score` + `steps_to_end`). The existing
  `StepRecord` is lossy (drops `target_id`/`amount`/`metadata`), so storing the
  full action JSON is what makes cheap offline re-expansion possible.
  `verify_capture_roundtrip` is the gate: re-simulating the log reproduces the
  captured per-step (score, ante, money) exactly, and that survives JSON
  persistence. 8 tests in `tests/test_ml_dataset.py` green (round-trip exactness,
  JSON-persisted round-trip, multi-phase coverage incl. shop/pack actions,
  faithfulness to `generate_trajectory`, determinism); encoder's 13 tests still green.
- **Step 0.3 DONE — value net + training harness** (`src/balatro_ai/ml/model.py`,
  `ml/train.py`). `ValueNet` is a set-encoder: embeddings for joker/card/shop/boss/
  consumable/voucher identities → masked mean-pool over the variable-length sets →
  concat with the fixed scalar/level/deck blocks → MLP trunk → win-prob logit,
  sized from `encoding_spec()`. `collate_states` pads `EncodedState` sets into
  tensors with masks; `train.py` is a deterministic BCE trainer with an eval split,
  versioned checkpoint save/load, and `overfit_check`. Gate passes: the net fits a
  tiny synthetic set (label = joker set contains "Blueprint", money uninformative)
  to **loss 0.000 / 100% accuracy**, and checkpoints round-trip. 6 torch-gated tests
  in `tests/test_ml_train.py`; torch 2.12.0+cpu installed (already in the `ml`
  extra). Full ml suite **27 tests green** (13 encoder + 8 dataset + 6 train).
  **Stage 0 (foundations) is COMPLETE.**
### 2026-05-31 — Stage 1: learned value head works; first play-leaf A/B faster-but-worse

- **Stage 1.1 + 1.2 DONE — bootstrap dataset + learned value head.**
  `src/balatro_ai/ml/bootstrap.py` generates teacher captures in parallel
  (resumable JSONL) and re-expands them to examples (2 tests green). Generated a
  **512-run `basic_strategy_bot` dataset** (61 wins, 11.9%, 76k examples). After a
  64-run cut overfit hard (val win-AUC 0.37), `ValueNet` gained an **expected-ante
  head** (dense target) + dropout/weight-decay. On a held-out-**by-run** split the
  **ante head generalizes (val corr 0.43)**; the binary-win head barely does (val
  AUC 0.58) — the dense ante target is the learnable signal. 29 ml tests green;
  checkpoint `phase8_value_v0.pt`.
- **Stage 1.3 first A/B (12 seeds, v2 beam) — learned leaf faster but worse; gate
  NOT met.** `ml/leaf.py::ValueNetLeaf` (ante head) ran **~2× faster** per
  play-decision (242ms vs 476–488ms for the rollout leaves) but at **lower
  quality** (mean ante 3.67 vs 5.2–5.4; median 2.5 vs 5.5–6). Diagnosis: the ante
  value is a coarse *whole-run* signal — wrong granularity for *within-blind* play,
  where `clear_probability` directly measures blind-clear — plus distribution shift
  (trained on teacher states, queried on beam lines). No production code touched
  (injected via `SolverPolicy(play_policy=...)`; A/B in `scripts/phase8_leaf_ab.py`).
- **Stage 1.3 Option A (rollout distillation) — works, nearly closes the gap.**
  `ml/distill.py`: `CollectingClearLeaf` records `(state → clear_probability)` from
  the v2 beam's own leaf states (fixes distribution shift) labeled with the rollout
  output (fixes granularity); `train_distill` regresses `ValueNet.clear_head`. On 16
  collection seeds (~16.5k pairs) the net reproduces the rollout at **val corr 0.87**
  (held-out seeds; corr 0.48@2 seeds → 0.87@16, so data-scalable). A/B (12 seeds, v2
  beam): the distilled clear-leaf is **~2.2× faster** per play-decision (224ms vs
  489–505ms) and recovered quality from the ante-head's 3.67 to **mean ante 4.67** —
  but still ~0.5–0.75 antes short of the rollouts (5.17–5.42). So *faster and
  almost-equal*, not yet a clean "faster AND ≥quality" pass; the gap is small and
  data-closeable. 31 ml tests green; `ml/leaf.py::ValueNetLeaf(head="clear")`,
  checkpoint `phase8_clear_v0.pt`. No production solver code touched.
- **Scaling distillation BACKFIRED (16→48 seeds): beam ante 4.67→3.67.** MSE
  distillation of the mean-heavy rollout labels (clear-prob mean ~0.92) regresses to
  the mean — `pred_mean` 0.904→0.945, under-fitting the RARE low-clear leaves that
  actually drive play choices. More data amplified the collapse. Lesson: the distilled
  leaf needs a **loss fix** (reweight/rank the low-clear states, or cleaner labels),
  NOT more data; `phase8_clear_v0.pt` (16-seed, ante 4.67, ~2.2× faster) stays the best
  distilled leaf.
- Decision point: Option A is validated-but-finicky and is a **SPEED-only** win (capped
  at rollout quality). The strength path to superhuman is **Stage 2 (policy head)** —
  imitate teacher actions (the bootstrap captures already hold `(state, action)` pairs)
  to prune/guide search AlphaZero-style, then self-play. **Recommended next: Stage 2.**
  (The Option-A reweight fix is a known fallback if a fast leaf becomes the bottleneck.)

### 2026-05-31 — Stage 2: policy head (type policy strong; per-card policy failed)

- **Policy head built + imitation-trained** (`ml/policy.py`, `ValueNet.policy`): a
  14-way action-TYPE head + a per-card play POINTER (per hand position, since the
  trunk pools the hand). Trained on the existing 512-run captures (no new data-gen).
  Held-out (by-run): **action-type acc 0.753 vs base rate 0.233** — a strong,
  generalizing signal (train 0.768, minimal gap). But the **per-card pointer FAILED**:
  card_pos_acc 0.564 (≈ chance), subset-exact 0.8%. Card selection is combinatorial
  (which SUBSET forms the best hand), so position-independent BCE is the wrong model;
  it needs a candidate-SUBSET scorer (a pointer over the enumerated candidate plays).
  32 ml tests green; checkpoint `phase8_policy_v0.pt`.
- Pattern across Stage 1–2: **coarse targets learn** (ante 0.43, type 0.75, clear-prob
  0.87); **hard/combinatorial/discriminative targets need better modeling** (win,
  exact cards, low-clear states). The forward model + learning pipeline are solid;
  each head needs its right architecture.
- **Hand-type play policy ALSO failed** (`hand_type_head` predicting which of 12 poker
  hands the teacher plays): held-out hand-type acc 0.519 vs base 0.479 (+0.04, ≈ base).
  **ROOT CAUSE (now clear): the play decision is dominated by the SPECIFIC dealt hand**
  ("what's the best hand I can form?"), but the set-encoder **pools the hand away** — so
  neither the pooled trunk (hand-type head) nor position-independent per-card scoring
  can capture it. The action-TYPE policy works precisely because type is phase/coarse-
  state-driven, NOT hand-specific. A real play policy must score **enumerated candidate
  plays** with explicit per-candidate (hand-type + cards) features — a candidate-subset
  pointer, not a state-only head. Documented negative; `phase8_policy_v1.pt`.
- **Map after Stage 1–2** — WORKS: ante value (0.43), distilled clear-leaf (corr 0.87,
  ~2.2× faster), action-type policy (0.75 vs 0.23). FAILS via the pooled rep: win head,
  per-card / hand-type play policy. Clear next: (a) **candidate-subset play policy**
  (enumerate + score candidates) for a real play prior; then (b) wire value + priors
  into a neural-guided search + self-play loop. Per `PHASE8_NEURAL_PLAN.md`.

### 2026-06-01 — Stage 2.2: candidate-subset play policy WORKS (first play-side win)

- **`ml/play_policy.py` + `ValueNet.play_candidate_scores`**: score each *enumerated*
  candidate play (pooled subset card-embeddings + hand-type + size + global context),
  trained by negative-sampling CE to rank the teacher's chosen play above random
  subsets — the corrected model that sidesteps the pooled-state problem.
- **Held-out: top-1 0.388 vs random baseline 0.031 (~12.5×), train≈val (0.398/0.388 —
  generalizes cleanly).** The FIRST play-side head to learn: scoring enumerated
  candidates cracked the combinatorial play decision the pooled-state heads couldn't.
  (top-1 0.39 = a strong PRUNING prior — top-k recall is high — not a perfect #1
  ranker.) 33 ml tests green; checkpoint `phase8_playpolicy_v0.pt`.
- **All three neural-guided-search components now exist:** value leaf (distilled
  clear-prob, ~2.2× faster), action-type prior (0.75 vs 0.23), play-candidate prior
  (0.39 top-1 / 12.5× random). Next: wire them into a **pruned neural-guided beam**
  (policy → top-k candidates, value leaf → eval) and A/B vs the heuristic beam; then
  the self-play loop. Per `PHASE8_NEURAL_PLAN.md`.

### 2026-06-01 — Stage 2.3: neural-guided beam wired + A/B → play-policy pruning is a DEAD END

- **Wired with NO production solver edits** (`ml/neural_search.py`): the v2 beam already
  abstracts `candidate_provider` + `leaf_evaluator`, so `PolicyCandidateProvider`
  (ranks legal plays by `play_candidate_scores`, returns top-k) + `ValueNetLeaf(clear)`
  inject straight into `SearchV2PlayPolicy` via `SolverPolicy(play_policy=...)`.
- **10-seed A/B (`scripts/phase8_neural_search_ab.py`), depth 3 width 2** — clean
  decomposition of the two neural changes:
  | condition | mean ante | ms/decision |
  |---|---|---|
  | heuristic (TopK + rollout) | **5.1** | 494 |
  | leaf_only (TopK + distilled clear leaf) | **4.5** | **220 (2.2× faster)** |
  | neural_full (policy-prune + clear leaf) | **3.1** | 301 |
  - **Distilled leaf: −0.6 antes for 2.2× speed** — a real, usable data-gen accelerator.
  - **Policy candidate-pruning: −1.4 antes AND slower than leaf_only** → strictly
    dominated. The learned play-policy is a WORSE candidate ranker than the cheap
    immediate-score heuristic.
- **WHY (118-state probe, `scripts/phase8_policy_provider_probe.py`):** the policy is
  **anti-correlated** with immediate-score (mean rank-corr **−0.68**, agree@1 = 0.00) —
  NOT a wiring bug (rankings are structured/consistent: it always tops a full 5-card
  hand). The set-encoder pools the hand away, so the policy learned a *coarse* "prefer
  complete hands" prior, while immediate-score encodes resource-aware efficiency
  (clear the blind with minimal cards, conserve hands/discards). Hard-pruning to the
  policy's top-k discards the specific good plays the heuristic surfaces → runs derail.
- **Root structural reason it can't win:** the play policy was *distilled from the
  heuristic-fed beam*, so its whole training signal lives INSIDE the immediate-score
  candidate set — it's a lossy compression of the ranker it's trying to beat. The
  heuristic play ranker is already near-optimal AND cheap; there is almost no headroom
  on the play side. **Conclusion: play-selection is the wrong place for the neural net.**
  The bot dies ~ante 5–6 with strong early play → the lost winrate is a BUILD/economy
  problem (jokers too weak late), i.e. the SHOP, where heuristics are crude and a
  learned value fn has real headroom. 34 ml tests green. Leaf is bankable for speed.

### 2026-06-01 — Stage 2.4: value-guided shop FAILS → the ENCODER is the shared bottleneck

- **Added a clean injection seam** (`SolverPolicy.shop_leaf_value_fn`, the economy-side
  analog of `play_policy=`): a factory `(root_state) -> (leaf_state -> float)` passed to
  `best_shop_action`'s existing `leaf_value_fn`, taking precedence over the archetype
  leaf. `ml/shop_value.py`: the value head (`ante`) as the shop leaf, z-score calibrated
  to the heuristic leaf's scale (so `leaf_weight=0.35` stays balanced — a fair swap).
- **16-seed A/B (`scripts/phase8_shop_value_ab.py`), v2 d3 w2:**
  | condition | mean ante | winrate |
  |---|---|---|
  | heuristic shop | **5.6** | **18.8%** (3/16) |
  | neural-value shop | **1.9** | **0%** (0/16) |
  Catastrophic. **Calibration exposed the cause: the value head's output std is ~0.05
  on [0,1] — it predicts ~0.8 for nearly EVERY shop state** (near-constant; can't
  discriminate).
- **Decision-mix probe (`scripts/phase8_shop_value_probe.py`, identical states):** the
  neural leaf picks **SELL 17/30** vs the heuristic's 3 — it **liquidates its own
  jokers**. Mechanism: selling = −joker (pooled away → invisible to the head) +money
  (a scalar feature the head DOES see) → the post-sell state scores ≥ → it sells the
  build → dies ante 2. agreement with heuristic 0.30.
- **THE CONVERGENT FINDING (3 independent failures, 1 root cause):** play-candidate
  policy (coarse, anti-correlated −0.68), win head (never learned), and now the shop
  value head (near-constant, sells jokers) ALL fail because the **DeepSets mean-pool
  set-encoder averages jokers/cards/shop into a blur**, destroying the per-item
  information that play discrimination AND economy valuation both need. The ONE head
  that worked (candidate-subset play policy, 0.39) worked precisely because it fed the
  model EXPLICIT per-candidate features instead of the pooled rep — the proof that
  per-item info is the fix. **The bottleneck is the trunk/encoder, not which head.**
- **NEXT (proposed): rebuild the encoder with ATTENTION over the joker/card/shop sets**
  (transformer-style, no mean-pool) so per-joker identity/value is visible, then retrain
  the value head and re-test the shop leaf + play side. This is the foundational unlock
  (the AlphaZero analog: a too-weak encoder caps every downstream head). 34 ml tests
  green. The `shop_leaf_value_fn` seam + calibration harness are reusable for the retest.

### 2026-06-01 — Stage 2.5–2.6 + Option A: encoder ruled out, rollout shop fails, but the value ceiling is LABEL NOISE (fixable)

- **Stage 2.5 — attention encoder RULED OUT.** Built `ValueNet(encoder="attention")`
  (self-attention over joker/card/shop tokens + CLS + n_query attention-pool, replacing
  mean-pool; default-off, rebuildable via hparams) + `scripts/phase8_encoder_validate.py`.
  A 2-epoch smoke looked great (attention joker-removal 0.92 vs 0.58) but was an
  **under-trained fluke**. Decisive 512-run (8× data): attention ante-corr **0.448 vs
  mean 0.498** (attention *worse*), val-loss identical, and BOTH fail joker-removal in the
  *wrong* direction. Both architectures hit a ~0.47–0.50 corr ceiling at both data
  sizes → **the encoder is not the lever; the data/target is.**
- **Stage 2.6 — rollout shop FAILS (live).** `search/rollout_shop.py` `RolloutShopPolicy`
  (forward-model rollout per candidate). 2-seed A/B: ante 1.5 vs heuristic 5, **12s/shop
  decision**. Diagnostic (not a bug): at an ante-1 shop ALL candidates roll out to the
  same value (horizon cap) — early antes are trivial → zero discrimination. **Play-rollout
  works because play value is SHORT-horizon (clear this blind); shop value is LONG-horizon
  (win the run)** — a short rollout saturates, a full one is too slow. That's why the
  heuristic's hand-crafted scaling-value is hard to beat.
- **Option A — the value ceiling is LABEL NOISE, not bias (first POSITIVE result).**
  `scripts/phase8_rollout_value_diagnostic.py` tests whether the joker-value signal is in
  the TARGET via PAIRED rollout joker-removal (same seeds with/without each joker, +3-ante
  bounded, re-derived legal_actions, parallel). **16 ante≥4 states / 78 jokers / 6 samples:
  rollout-target joker-removal Δ = +0.383 antes, 70.5% positive** vs the **net (single-traj
  labels) Δ = +0.012 / 56% (flat).** The signal *exists* in the data; the value head was
  flat only because single-trajectory final-ante labels are too high-variance to extract it.
  ⇒ the ~0.47 ceiling is a LABEL-NOISE ceiling.
- **Meta + pivot:** four neural attempts lost to the heuristics+forward-model (play-policy,
  shop-value-net, encoder, rollout-shop) — the net is premature as a value SOURCE; its
  proven role is distilling rollouts (accelerator). Per the AlphaZero-from-no-data analysis,
  the engine (forward-model search) is the prerequisite that generates self-play data. **NEXT
  = Option A Part 2:** relabel states with MULTI-ROLLOUT-AVERAGED values (low variance),
  retrain the value head, confirm joker-removal Δ goes 0.01 → clearly positive + corr beats
  0.47. The averaged-rollout labeler then *is* the AlphaZero value-data generator. 34 ml
  tests green.
- **Quality (not just count) probe** (`scripts/phase8_joker_quality_probe.py`): the +0.38
  removal signal could be the trivial "more jokers > fewer" (count). To isolate QUALITY,
  measured the split-half reliability of the *demeaned* per-joker Δ (subtract each state's
  mean → removes count, leaving relative joker quality). **14 DIVERSE builds (1 state/seed),
  68 jokers, K=10: demeaned split-half corr = 0.66**, with build-specific carries (Stuntman
  +1.39 in one build, Abstract Joker +0.42 in another, Spare Trousers +0.73 in a third). (A
  one-build sample inflated this to 0.95 — fixed by capping states/seed.) ⇒ averaged rollouts
  carry BOTH count AND real build-quality signal → Part 2 can plausibly *beat* the heuristic,
  not just stop the joker-selling. GREEN LIGHT for Part 2.

### 2026-06-02 — basic_strategy audit + A/Bs: heuristic polishing is CONCLUSIVELY tapped; the wall is build power

- **4-agent fresh-eyes audit** of `basic_strategy_bot` (play, shop, build-valuation,
  orchestration). Implemented the top findings, each **env-gated (default off) + A/B'd**:
  - **#1 Blueprint valuation bug** (`build_scoring._best_order_sample_score`,
    `BALATRO_COPY_JOKER_SCORING`): `_jokers_after_buy_for_scoring` appended candidates
    last, so Blueprint/Brainstorm (copy the joker to their RIGHT) copied NOTHING → valued
    at flat base. Fix searches the copy-joker's best position. **A/B 100 seeds: 13 vs 14
    baseline — neutral.**
  - **#4 planets aren't credited as scaling** (`build_profile._hand_level_scaling_score`,
    `BALATRO_PLANET_SCALING`): a leveled hand read as "no scaling." **A/B: 9 vs 12 —
    neutral/slightly negative** (over-credits planets → under-buys scaling jokers).
  - **#2 `safe_margin` builds-less-when-safe** (`profile.safe_margin`,
    `BALATRO_NO_SAFE_MARGIN`): raised buy/cost/reroll/interest thresholds when ahead.
    **A/B: 12 vs 12 winrate — neutral, but +4 to ante 8.**
  - **Campfire mis-classified as a scaler** (`BALATRO_RESET_JOKER_DISCOUNT`): it's in
    SCALING_JOKERS+XMULT_JOKERS but RESETS each boss + needs a sell-economy (only viable
    ante 6+). Discount its scaling/xmult role credit ×0.15 before ante 6 (verified: ante-1
    Campfire scaling 32→4.8, xmult 30→4.5). Correct fix; expected-neutral (rare).
  - **#7 shop-audit perf gate** (`config.shop_audit_enabled`, default **on** =
    behavior-preserving): the per-option planner re-run (~2× late-shop cost) is skipped
    when off; turned **off in the rollout pilot** (`phase8_value_relabel_retrain`) for
    faster label-gen. Pure perf, no winrate effect.
- **NEW play-logic feature — dig-via-play** (`blind_solver._dig_via_play_action` +
  `draw_evaluation._best_flush_dig`, `BALATRO_DIG_VIA_PLAY`): when greedy can't clear (a
  guaranteed loss) and discards are gone, **use spare HANDS as discards** — play
  throwaways to dig for a clearing flush over multiple hands (deck-order-aware: known →
  cumulative `known_deck[:budget]` window, unknown → hypergeometric). Found via early-loss
  audit (`scripts/phase8_early_loss_audit.py`): seed 2 died ante 1 (254/600) playing Two
  Pair greedily when a multi-hand flush dig clears (5th heart at draw position 11; even a
  fully-debuffed flush scores 490 with Droll). Fires only in guaranteed-loss spots so it
  can't regress. **Verified: seed 2 ante 1→3 with dig on. A/B 100 seeds: 13 vs 14 —
  winrate-neutral, but ante-1 deaths 1→0, ante-8 reached 21→23, loss-frac 0.81→0.78.**
- **THE CONCLUSIVE FINDING:** all four fixes + the dig reshape the ante distribution
  (further, fewer early deaths) but are **winrate-neutral**. The bot **reaches ante 8
  ~21–23% but wins ~12–14%** — that ~8–10% gap is runs that reach the final boss and
  **lose to it**. The wall is *clearing ante 8* = pure end-game **build power**. Heuristic
  play/shop polishing is definitively tapped (now proven 4 ways + the dig). **The only
  lever is making builds genuinely stronger by late game → the value-function/search
  direction**, not heuristics. (Also confirmed ~1-win baseline wobble from Python hash
  randomization → pin `PYTHONHASHSEED=0` for exact A/B baselines.)

### 2026-06-02 — Phase 8 encoder was BROKEN (3 bugs); value lever is DATA not rollouts; play-policy unlocked

- **Audited the encoder** (degenerate-bucket rate over 1164 real states) — the "encoder
  RULED OUT" verdict (Stage 2.5) had been reached on a **broken encoder**. Found + fixed
  THREE bugs, each tested + committed; `ENCODING_VERSION` 1→3:
  - **Suit/rank mis-encoding** (`encoding._canon_suit`/`_canon_rank`, e103b99): real sim
    cards use single-letter suits ("S"/"H"/"D"/"C") and "T" for ten, but the vocab only
    knew "Spades"/"10" → EVERY card's suit → SUIT_NONE, deck suit-counts all 0, tens →
    rank 0 ("2"). The net saw **no suit info**; `_classify_hand` never detected a flush.
  - **Packs/vouchers unencoded** (`_shop_card_dicts` merge + `boosters` vocab, c7b24d4):
    booster packs live in `modifiers["booster_packs"]`, the voucher offer in
    `["voucher_cards"]` — separate fields the encoder never read (only `shop_cards`).
    Packs were **completely invisible** (no type/cost/identity).
  - **Joker scaling counter always 0** (`_joker_counter` reads metadata, 359d3ae):
    `joker.effect.*` is None in the local sim; the live value is in
    `metadata['current_mult'|'current_xmult'|'current_chips']`. The net couldn't tell a
    +50 Ride-the-Bus from a fresh one. (Stickers don't appear on white stake.) 6 new
    regression tests; **45 ml tests green**.
- **Clean re-tests on the fixed encoder** (reuse `phase8-bootstrap-basic.jsonl`, replay +
  re-encode, identical hyperparameters — only the encoder changed):
  - **Value head (bootstrap, single-traj): val ante_corr 0.43 → 0.48. MODEST** — the
    encoder was *not* the value-head's binding constraint.
  - **Play-candidate policy: val top1 0.388 → 0.533 (+37%). BIG** — the fix unlocked
    suit-aware play evaluation (the subset scorer sees flushes/straights now). Per-card
    pointer stayed flat (DeepSets mean-pool blur, a separate known issue). Policy heads
    have *clean* labels (teacher's action) → they benefit from the encoder where the
    noisy-label value head couldn't.
- **Option A Part 2 (rollout-averaged value relabel — the planned "label-noise" fix)
  FAILED.** 1500 states × 5 full-game rollouts (≈6h) → **val_corr 0.10**, *worse* than the
  cheap bootstrap's 0.48. Root cause = **data starvation**: clean rollout labels cost ~50×
  more per label → only 1200 train states vs the bootstrap's 61k → the net underfits to a
  near-constant (label std 0.82; joker frac_pos 0.99 but Δ 0.0004). The averaged-target
  signal is real (Part 1), but you can't *train* a net on 1500 of them. **Rollout-labeling
  is a dead end at feasible scale** (60k states × 5 ≈ 60+ h).
- **Revised map of the value-head lever:** not the encoder (modest), not rollout-relabel
  (data-starved dead end) — it's **data quantity + regularization on the cheap bootstrap**
  (single-traj, 76k examples nearly free via replay; overfits train 0.73 / val 0.48).
  Checkpoints: `phase8_value_v3_bootstrap.pt` (0.48, best value), `phase8_playpolicy_v3.pt`
  (0.53, good play prior), `phase8_value_relabel_v1.pt` (0.10, weak). **NEXT:** shop A/B
  with the bootstrap-v3 head (does 0.48 beat the heuristic shop?), then scale the bootstrap
  (512→2000 runs + stronger regularization) to push val_corr past 0.48.

## Next Steps

> **Top priority (2026-05-31): the Phase 8 neural build** — `PHASE8_NEURAL_PLAN.md`.
> Next concrete work is Stage 0.2 (data pipeline: extract `(encoded_state, action,
> outcome)` per step by re-simulating action logs) → Stage 0.3 (minimal torch
> trainer) → Stage 1 (learned value leaf, bootstrapped from `basic_strategy_bot`).
> The numbered items below are now secondary solver/RNG threads.

1. Continue Phase 4 toward the native solver. Target: `solver_beam_play_action_native(state, depth, width) -> Action` that runs the whole beam in Rust. See `RUST_PORT_PLAN.md` §6 (Phase 4 spec).
2. Close the remaining narrow RNG edge if needed: Illusion-generated shop playing cards carry global `math.random` into first Buffoon pack selection; optional Overstock slot-count fixtures can be added afterward. Current validation commands are `python -m balatro_ai.rng.validate_surfaces --all`, `python -m balatro_ai.rng.validate_shop_sequence --all`, `python -m balatro_ai.rng.validate_spectral_helpers --all`, and `python -m unittest discover -s tests -p "test_rng*.py"`.
3. **Raise the `SolverPolicy` data-gen winrate** (the active priority — see "In Progress" and memory `project_datagen_speed.md`). The solver is built (`solver/policy.py`, M1-M5.5 in `SOLVER_PLAN.md`) and generating data; it's at ~8% after the churn fix. Next: hunt more systematic leaf/action-value mis-valuations with `scripts/shop_decision_trace.py` (the method that found the churn + play-value wins), and improve build SCALING (xmult/synergy valuation). Do NOT chase leveling-term tuning or deeper shop search (both measured inert/negative). Paired A/Bs (`shop_paired_ab.py`, ≥80 seeds, `play_width=1` to match data-gen) are the validation gate.
4. Once the solver winrate is acceptable, do a 10k+ seed data-generation run (prototype on ~10 seeds first). The Rust core makes this far cheaper — a 10k-seed run estimated at ~50 hours becomes ~10-15 hours once Phase 4 lands.
5. Once a trajectory dataset exists, pivot to Phase 8: imitation learning on policy + outcome-based value head.
6. The two deferred minor sim bugs (Drunkard sell, Credit Card reroll) can be fixed any time — they take the audit from 99.9% to 100%.
7. Live-bot improvement (`basic_strategy_bot` tuning, `search_bot_v2` experiments) is de-prioritized but not banned; treat it as the second-class path. Same-seed A/Bs still required if any change lands.
