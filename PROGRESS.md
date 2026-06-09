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

### 2026-06-03 — Option C increment 1 (TD(λ) targets): value head breaks the ceiling on-distribution, but DISTRIBUTION SHIFT blocks the shop

- **TD(λ) relabel of the bootstrap captures** (`scripts/phase8_td_relabel.py`): replace
  high-variance MC final-ante labels with TD(λ) targets `G_i=(1-λ)·V(s_{i+1})+λ·G_{i+1}`
  (γ=1, terminal `final_ante/9`), bootstrapped from the bootstrap-v3 net; fitted value
  iteration (freeze V, relabel, train fresh net, swap). Reuses captures (replay + cache),
  no rollouts — cheap.
- **A SINGLE TD step beat the MC bootstrap** (first ceiling break this session): val
  ante_corr **0.48 → 0.58**, val shop_std on the basic_strategy training states **0.074 →
  0.10**. λ=0.5 ≈ λ=0.9 at iter 0. The variance-reduction thesis held — lower-variance
  targets let the net fit the outcome better AND discriminate shop states.
- **Naive fitted value iteration COLLAPSES**: iters 1→3 degrade both (corr → 0.52–0.56,
  shop_std → 0.07–0.08, train_target_std shrinks each round) — bootstrapping off a flat-ish
  V pulls variance toward the mean. Stable iteration needs a target network / regularization.
  The win is the single relabel step; banked `phase8_value_td_iter0.pt` (corr 0.575).
- **Shop A/B (100 seeds) STILL FAILS: TD-iter0 = 2.64 ante / 0% vs heuristic 4.87 / 5%**
  (≈ bootstrap-v3's 2.61 / 1%). SMOKING GUN: calib shop_std on the DEPLOYMENT (SolverPolicy)
  states = **0.073 ≈ bootstrap-v3's 0.074** — vs 0.10 on the basic_strategy TRAINING states.
  **DISTRIBUTION SHIFT**: the head discriminates the shop states it trained on but is just as
  flat as before on the SolverPolicy shop states the beam actually visits.
- **CONCLUSION — offline shortcuts are exhausted.** MC bootstrap (flat shop), rollout-relabel
  (data-starved), TD-relabel (great on-distribution, flat off-distribution) ALL fail to guide
  the shop. The value head CAN learn (corr 0.58) but training≠deployment distribution + shop's
  long-horizon credit assignment block it. The principled fix is the **ON-POLICY loop** (the
  AlphaZero analog the plan always pointed to): generate states + search-improved targets with
  the SAME search that deploys the head, train on that distribution, iterate. NEXT (await user):
  on-policy loop's first increment (capture SolverPolicy runs → TD-relabel → shop A/B) vs the
  full iterative loop.
- **Shop action-ranker prototype (`scripts/phase8_shop_action_label.py`) — GATE FAILED →
  shop-value CONCLUSIVELY CLOSED.** Per the "train Q/ranking, not V(state)" synopsis: at shop
  states, enumerate legal actions, apply each with the forward model, evaluate every branch via
  COMMON-RANDOM-NUMBER bounded rollouts (paired delta cancels shared variance). Gate = does the
  action ranking REPRODUCE (split-half) + show headroom over the heuristic? Bounded M=10:
  top1_stable 0.30, half_corr 0.24, reliable_disagree 0.033. Rigorous M=16 + coarse-V truncation
  (TD-iter0 head estimates the tail): top1_stable 0.27, half_corr 0.18, reliable_disagree 0.033 —
  V-truncation slightly HURT (injected the value head's own noise). VERDICT: even at the best
  feasible config the CRN signal is noise-dominated (the shop edge is a residual *below
  feasible-rollout resolution*) AND the heuristic shop SELECTION is already near-optimal (1/30
  reproducible disagreements). Every shop-value form (V-net, rollout, TD, action-ranker) is now
  closed. **KEY IMPLICATION: shop selection is near-optimal yet winrate is 12-14% → the gap is
  NOT selection; it's build-CEILING, PLAY, or META.** NEXT: ante-8 build-vs-play diagnostic (for
  ante-8-loss states, can a deep play search clear the hand? = play-limited vs build-limited) +
  audit discrete meta decisions (skips/tags, vouchers, pack picks). Play was "tapped" only on
  AVERAGE — endgame play untested.

### 2026-06-03 evening — Refresh/audit pass; pack-aware labeler correction; SolverPolicy exposed as strongest current bot path

- **Phase 1 cleanup after encoder fixes:** tightened suit normalization so exact aliases
  (`S`, `spade`, `Spades`) still work but unknown/suitless labels such as `Stone` stay
  `SUIT_NONE`; made the XMult counter semantics explicit (`current_xmult=1.0` is a visible
  baseline value); updated runnable Phase 8 script defaults/examples to v3 checkpoints; and
  made stale-checkpoint errors include the path/remedy. Focused ML tests + compile pass green.
- **Corrected the shop action-ranker gate:** the prototype omitted `OPEN_PACK`, so prior
  headroom/agreement conclusions under-tested one of the most important shop actions.
  After adding pack actions and tie-aware metrics, a small pack-aware rerun
  (`8 states x 4 CRN rollouts, +3 ante`) showed:
  `top1_stable=0.75`, `half_corr=0.058`, `reliable_disagree=0.0`,
  `heuristic_within_0.05=0.875`, `mean_best_margin=0.0`.
  Revised verdict: pack omission was a real measurement bug, but the corrected cheap gate
  still does **not** produce a useful shop-ranker target; most "disagreements" are ties/noisy
  residuals, not clear regret.
- **Death-margin diagnostic hardened:** now buckets only true `RUN_OVER` losses and reports
  no-action/sim-error/max-step aborts separately, so aborted simulations cannot masquerade as
  death-margin losses.
- **Current winrate path:** exposed `SolverPolicy` through the bot registry as
  `solver_policy_bot` (and GUI list) so the stronger offline-solver policy is benchmarkable via
  normal tooling. Same generic sim canaries with `PYTHONHASHSEED=0`:
  `basic_strategy_bot` 2/24 (8.3%, 177s), `solver_policy_bot` 4/24 (16.7%, 309s).
  On 12 seeds: basic 1/12, search_bot_v2 2/12, solver_policy_bot 3/12. This is slower but is
  the best current explicit bot path. Tested likely knobs: search_bot_v2 hand width 1 regressed
  (1/12), shop depth 2->3 regressed (3/12 -> 0/12), and build-aggression+boss-aware remained
  negative (3/12 -> 0/12). Leave defaults alone; use `solver_policy_bot` as the current
  quality baseline.

### 2026-06-03 late evening -- paired A/B harness; solver shop memory; best current path is solver shop + basic play

- **Added `scripts/bot_paired_ab.py`** for same-seed bot A/Bs. It runs both bots on each
  seed inside the same worker, disables shop audit payloads, records aborts separately, and
  reports paired ante/win flips. It now avoids misleading aggregate score deltas by only
  aggregating score on same-ante losses and by reporting normalized loss-fraction deltas.
- **Found and fixed a real `SolverPolicy` shop-context bug:** full solver shop search was
  called with a fresh `ShopSearchContext()` every decision and no protected-joker memory,
  unlike `search_bot_v1/v2`. It now tracks rerolls, packs opened, last-slot fills, and
  newly bought protected jokers across the current shop, preserves that state through
  `BOOSTER_OPENED`, and resets on non-shop/new-shop transitions. Focused tests cover the
  context handoff and memory reset behavior. Solver-owned shop actions are also mirrored
  into the BasicStrategy fallback's shop memory, so fallback pack-card choices see the
  correct pack-open context after solver shop search opens a pack.
- **Measured full `solver_policy_bot` after the shop-memory fix:** on 24 paired seeds vs
  `basic_strategy_bot`, wins were 3/24 vs 2/24 and mean ante delta was only -0.17
  (better 9 / worse 8 / same 7). This is an improvement over the prior full-solver
  36-seed shape (8/36 wins but mean ante -0.56), but the full solver is still spiky and
  much slower (~92s/run vs ~36s/run on the 24-seed run).
- **Traced the remaining early solver collapses:** the problem is not just shop selection.
  Full solver play search makes fragile early blind decisions on some seeds (e.g. ante-1
  deaths after search-driven discards/weak build pieces), while basic play often survives
  those same situations. The full solver's wins and baseline's wins also barely overlap,
  so treating full solver as a straight replacement is premature.
- **New best measured experiment lane:** added `solver_shop_basic_play_bot`, which uses
  `SolverPolicy` shop search but delegates play/fallback decisions to `BasicStrategyBot`.
  Same-seed A/B vs `basic_strategy_bot`:
  - 24 seeds: hybrid 5/24 wins, mean ante 6.67 vs 5.62, paired d_ante +1.04 median +1,
    better 15 / worse 2 / same 7, CPU ~47s/run vs ~36s/run.
  - 36 seeds: hybrid 7/36 wins, mean ante 6.61 vs 5.86, paired d_ante +0.75 median +1,
    better 21 / worse 6 / same 9, CPU ~47.5s/run vs ~35.6s/run.
  This is the strongest current local-sim lane by ante distribution, but it loses the
  baseline's 4 wins on the 36-seed slice while creating 7 different wins. Treat it as
  a candidate, not a final default. It is exposed in the registry and GUI list.

### 2026-06-03 night -- hybrid shop ablations: keep depth 2 / reroll 8; width 3 is the fast near-equal option

- **Regression traces for baseline-win / hybrid-loss seeds** showed many solver-shop
  `SELL -> BUY` churn lines, but the simple fix "never take solver SELL" was too blunt.
  Added `allow_shop_sells=False` and a registry-only
  `solver_shop_basic_play_no_sell_bot` ablation. On the first 24 seeds it was positive
  vs baseline but worse than the existing hybrid: **4/24 wins, mean ante 6.17** vs the
  default hybrid's **5/24 wins, mean ante 6.67**. Conclusion: solver-only sells are
  sometimes harmful, but also create real wins; do not globally disable them.
- **Added SolverPolicy shop env knobs** for efficient tuning without new bot classes:
  `BALATRO_SOLVER_SHOP_DEPTH`, `BALATRO_SOLVER_SHOP_BEAM_WIDTH`,
  `BALATRO_SOLVER_SHOP_REROLL_SAMPLES`, and `BALATRO_SOLVER_SHOP_MIN_SEARCH_VALUE`.
  Tests cover the override path.
- **Shop horizon/width ablations on `solver_shop_basic_play_bot`:**
  - `depth=1` is bad: 24 seeds **0/24 wins, mean ante 5.21**, despite being faster.
    The hybrid needs two-step shop lookahead.
  - `depth=2,width=2` is positive but too weak: 24 seeds **4/24 wins, mean ante 6.12**.
  - `depth=2,width=3` is the best speed/quality compromise tested: 24 seeds **5/24,
    mean ante 6.54**, and 36 seeds **7/36, mean ante 6.56**, paired d_ante **+0.69**,
    better/worse/same **20/5/11**, CPU **~43.5s/run** vs width-4 default's **~47.5s/run**.
  - `depth=2,width=3,reroll_samples=4` loses the signal: 24 seeds **2/24, mean ante
    5.71**. Keep reroll samples at 8.
- **Negative shop-beam gate checked after seed-37 trace:** the beam sometimes accepts
  negative-value actions because default `min_search_value=-inf`. Thresholds improved some
  severe regressions but lost too much upside on the 24-seed slice:
  `min=0` -> **4/24, mean ante 6.42**; `min=-25` -> **4/24, 6.38**;
  `min=-60` -> **5/24, 6.33** vs default **5/24, 6.67**. Leave the default ungated; use
  the env knob only for diagnostics.
- **Current recommendation:** use `solver_shop_basic_play_bot` as the quality lane. Default
  width 4 is still the highest measured ante on the 36-seed slice (**+0.75 d_ante, 7/36
  wins**). For faster iteration, run with
  `BALATRO_SOLVER_SHOP_BEAM_WIDTH=3` (**same 7/36 wins, slightly lower ante, ~9% cheaper**).
- **Larger confirmation:** default `solver_shop_basic_play_bot` vs `basic_strategy_bot` on
  60 paired seeds: **12/60 wins vs 7/60**, mean ante **6.43 vs 5.78**, paired d_ante
  **+0.65** (median **+1.0**), better/worse/same **34/16/10**, CPU **~47.2s/run vs
  35.2s/run**. This is the strongest current measured lane and the first clearly meaningful
  local-sim winrate lift this pass, though it still loses 7 baseline wins while creating 12
  different wins.

### 2026-06-03 late night -- hybrid guard audit: 16/60 wins with funded-sell exception

- **Fixed a subtle guard bug before measuring:** the first `prefer_fallback_info_first_shop`
  implementation called `fallback.choose_action(state)` just to ask whether BasicStrategy
  wanted an info-first pack. That could mutate BasicStrategy's per-shop memory even when the
  solver ignored the fallback action. Replaced it with a pure reconstruction of
  BasicStrategy's info-first check using the same shop value helpers, then added regression
  coverage so the fallback is not called on that path.
- **Ablated the two hybrid guards on the same 60 seeds:**
  - no guards / old hybrid: **12/60 wins**, mean ante **6.43**.
  - info-first only: **14/60 wins**, mean ante **6.42**.
  - negative-sell fallback only: **12/60 wins**, mean ante **6.37**.
  - both guards before refinement: **16/60 wins**, mean ante **6.38**.
  The interaction is real: info-first creates extra upside, and the negative-sell fallback
  preserves two of those wins by preventing pointless `SELL -> BUY` sequences when the buy
  was already affordable or the sell was just search churn.
- **Found and fixed the downside pattern:** seed `0000052` regressed from ante 7 to ante 2
  because the negative-sell fallback blocked a sale that was actually needed to fund the
  planned visible joker buy (`Merry Andy -> Devious Joker`). The guard now falls back only
  for negative SELLs that do **not** fund the first planned BUY in the shop search path.
  Regression tests cover both cases: unfunded negative SELL falls back; funded negative
  SELL is allowed.
- **Refined current best:** `solver_shop_basic_play_bot` (info-first + negative-sell guard
  with funded-sell exception) vs `basic_strategy_bot`, 60 paired seeds:
  **16/60 wins vs 7/60**, mean ante **6.52 vs 5.78**, paired d_ante **+0.73** (median
  **+1.0**), better/worse/same **34/17/9**, CPU **~49.0s/run vs 35.5s/run**. This is now
  the best measured local-sim lane: same win count as the unrefined guard, better ante
  distribution, and the seed-52 floor regression fixed.

### 2026-06-04 after midnight -- stopping-point sweep: more ablations, no new default

- **Made the paired A/B harness interruption-safe:** an interrupted 60-seed reroll-sampling
  run exited without writing metrics, wasting the completed work. `scripts/bot_paired_ab.py`
  now writes an atomic partial metrics JSON after each finished seed pair, with
  `expected_n` and `complete` fields. This is a pure workflow/perf-safety improvement for
  long local runs under Codex/thread interruptions.
- **Remaining baseline-win losses traced:** the 7 baseline wins still lost by the hybrid are
  seeds `0000010`, `0000015`, `0000026`, `0000030`, `0000037`, `0000039`, `0000059`. The
  traces show no single clean correction: some are early first-shop ordering differences,
  some are build-archetype divergence, and some are later positive-valued sell/rebuild
  choices. This argues against another broad default guard without proof.
- **Tested two targeted guard ablations; both failed on 24 seeds:**
  - `solver_shop_basic_play_buffoon_bot` (force first-shop Buffoon when no jokers):
    **4/24 wins, mean ante 6.25** vs current default's **7/24, 6.25**.
  - `solver_shop_basic_play_sell_guard_bot` (fallback for unfunded sells while a joker
    slot is open): **3/24 wins, mean ante 6.38**. It removed some ugly sells but also
    killed too much upside. Leave both as registry-only diagnostics, not defaults.
- **Checked quality/sampling knobs after the refined default:**
  - `BALATRO_SOLVER_SHOP_BEAM_WIDTH=5`: **6/24 wins, mean ante 6.25**; not better than
    width 4.
  - `BALATRO_SOLVER_SHOP_REROLL_SAMPLES=12`: **5/24 wins, mean ante 6.46**, CPU ~59.7s/run;
    not worth it.
  - `BALATRO_SOLVER_SHOP_REROLL_SAMPLES=16`: promising on 24 seeds (**8/24, mean ante
    6.50**) but 60-seed confirmation was **15/60 wins, mean ante 6.60**, CPU **~66.7s/run**.
    This improves depth/ante distribution but loses one win and is ~36% slower than the
    refined default. It is a diagnostic/deep setting, not the quality default.
- **Current stopping point:** keep `solver_shop_basic_play_bot` default at depth 2,
  width 4, reroll samples 8, with the info-first and funded negative-sell guard. Best
  verified quality result remains **16/60 wins vs baseline's 7/60** at **~49s/run**.

### 2026-06-04 -- staged hybrid tested; 100-seed confirmation

- **Tested staged hybrids to recover baseline-win losses:** several lost-baseline traces
  diverged in ante-1/ante-2 shops, so two hidden aliases let BasicStrategy own early SHOP
  decisions before handing back to solver shop search:
  - `solver_shop_basic_play_basic_ante1_bot`: **3/24 wins, mean ante 5.67**. Hard miss;
    the solver's ante-1 shop aggression is carrying too much upside.
  - `solver_shop_basic_play_basic_ante2_bot`: **6/24 wins, mean ante 6.38**, then
    **15/60 wins, mean ante 6.28**. It loses fewer baseline wins (4 vs current default's
    7 on the 60-seed slice) but creates fewer solver wins and has lower overall winrate.
    Leave staged variants as diagnostics only.
- **100-seed confirmation for the current default:** `solver_shop_basic_play_bot` vs
  `basic_strategy_bot`, paired seeds `0000001..0000100`, `PYTHONHASHSEED=0`:
  **22/100 wins vs 13/100**, mean ante **6.52 vs 5.87**, median ante **7 vs 6**,
  better/worse/same **53/27/20**, win flips **21 for hybrid / 12 for baseline**, CPU
  **~56.0s/run vs 42.4s/run**. The lift survives the larger slice: +9 wins, +0.65 mean ante.
  Hybrid and baseline still overlap very little (only seed `0000097` won for both), so
  future gains likely need a real meta-decision or better shop value, not a single early
  fallback window.
- **Added a pipe-free subprocess backend to `scripts/bot_paired_ab.py`:** Codex's Windows
  sandbox denies `multiprocessing.Pipe`, so `--backend auto` now falls back from
  `ProcessPoolExecutor` to independent per-seed subprocess workers that write row JSON
  files. This preserves parallel paired A/B runs without requiring escalated Python
  prompts, while keeping the old process-pool path for normal local runs.
- **Tested a broader negative-action fallback diagnostic:** `solver_shop_basic_play_neg_action_bot`
  lets BasicStrategy override any negative-valued shop-search action when Basic has an
  active non-`END_SHOP` move. It looked plausible at 24 seeds (**6/24 wins, mean ante
  6.54**, better/worse/same **15/3/6**) but failed the 60-seed check: **13/60 wins,
  mean ante 6.33**, versus the current default's same-seed first-60 slice of **16/60 wins,
  mean ante 6.52**. It rescues some losses but kills too many current-default wins; keep
  it as a hidden diagnostic only.
- **Current quality lane remains unchanged:** `solver_shop_basic_play_bot` with depth 2,
  width 4, reroll samples 8, info-first guard, and funded negative-sell fallback. This is
  now verified at **22% on 100 local-sim paired seeds**, up from `basic_strategy_bot`'s
  **13%** on the same seeds.

### 2026-06-04 -- rare-hand commitment guard promoted

- **Found the next systematic loss shape:** seed `0000037` was a baseline win but hybrid
  died in ante 4 after overcommitting to an unsupported rare-hand/suit build
  (`The Family`, `Flower Pot`, `Sock and Buskin`). The existing support check treated a
  normal deck with four same-rank cards as "possible enough", so it did not penalize
  midgame commitment to Four of a Kind without rank manipulation or hand-level support.
- **Added a default-on rare-hand commitment reliability multiplier:** rare-hand payoff
  jokers such as `The Family` keep full value in ante 1-2, but from ante 3 onward their
  candidate/owned shop value is discounted unless the build has support from hand levels,
  rank-manipulation tarot cards, or extra duplicate-rank density. The weight remains
  ablatable via `BALATRO_RARE_HAND_COMMITMENT_RELIABILITY_W=0.0`.
- **Kept the early speculative upside:** the first version discounted ante 1-2 too, which
  rescued seed `0000037` but lost seed `0000053` by selling out of an early `The Family`
  line that later converted into a stable `Banner`/`Half Joker`/`Card Sharp` win. Gating
  the reliability multiplier to ante 3+ preserved both: seed `0000037` improves from
  ante 4 to ante 8, and seed `0000053` remains a win.
- **100-seed confirmation versus the saved current default:** the new default keeps
  `solver_shop_basic_play_bot` at **22/100 wins**, improves mean ante **6.52 -> 6.61**,
  and moves paired better/worse/same **53/27/20 -> 54/26/20**. All six changed solver
  seeds are non-win improvements: `0000027` ante 6 -> 7, `0000037` ante 4 -> 8,
  `0000046` same ante with higher score, `0000067` same ante with higher score,
  `0000076` ante 5 -> 8, and `0000094` ante 5 -> 6. Metrics:
  `.data/bot_paired_basic_solver_shop_basicplay_rarecommit_ante3_100.json`.
- **Current quality lane:** `solver_shop_basic_play_bot` with depth 2, width 4, reroll
  samples 8, info-first guard, funded negative-sell fallback, and ante-3+ rare-hand
  commitment reliability. It is still **22% on the 100-seed paired slice**, but with a
  better loss distribution than the previous default.

### 2026-06-04 -- first-divergence sweep; no second default from shop fallbacks

- **Generated a reusable first-divergence report:** `.data/first_divergence_100.json`
  compares `basic_strategy_bot` and current `solver_shop_basic_play_bot` on the saved
  100-seed paired slice, stopping at the first different decision. The main lesson is that
  the largest obvious category is not baseline-favorable:
  `ante 1, no jokers, Basic opens Buffoon, solver buys visible Joker` has **32 seeds**,
  but solver wins **6** and baseline wins **3**. The related
  `Basic opens Buffoon, solver buys planet` bucket is balanced (**17 seeds, 2 wins each**).
  This explains why broad Buffoon-first guards keep failing despite rescuing a few baseline
  traces.
- **Tested Stencil slot-discipline as an env-gated diagnostic:** added
  `BALATRO_STENCIL_SLOT_DISCIPLINE_W` to penalize ordinary joker buys that consume protected
  `Joker Stencil` slots. It improved seed `0000030` from ante 5 to ante 6, but the 25-60
  slice stayed **9/36 wins** and only one row changed (seed `0000030`). Leave it off by
  default.
- **Rejected two narrower fallback bots after A/B screens:**
  - `solver_shop_basic_play_buffoon_nonjoker_bot` only opens an ante-1 Buffoon over a
    non-joker buy. It lost the first-24 screen: **6/24 wins, mean ante 6.12** vs current
    default's **7/24, 6.25**, losing current solver win `0000005`.
  - `solver_shop_basic_play_neg_end_bot` falls back only when shop search chooses a
    negative-valued `END_SHOP`. It deepened some losses and gained seed `0000006`, but also
    lost solver wins `0000005` and `0000016`: **6/24 wins, mean ante 6.54** vs current
    default's **7/24, 6.25**. Not a winrate default.
- **Checked the high-precision one-joker planet pocket:** `solver_shop_basic_play_ante1_planet_bot`
  falls back only when ante-1 has exactly one joker, solver wants a planet, and Basic wants
  a joker or Celestial pack. It changed the intended traces but did not convert them:
  `0000026` still dies ante 8 and `0000081` still dies ante 8; the deliberately excluded
  standard-pack case `0000094` remains unchanged. Leave it as a diagnostic, not a measured
  default candidate.
- **Conclusion for this pass:** the rare-hand commitment guard remains the only new default
  from this round. The next likely real gains are not simple first-divergence fallbacks; they
  probably need either a better shop leaf value for late build quality or a learned/meta
  selector that can exploit the very low overlap between baseline wins and solver wins.

### 2026-06-04 -- follow-up audit: consumable scoring and ValueNet wiring

- **Fixed a stale diagnostic tool:** `scripts/shop_decision_trace.py` was still tracing a raw
  `SolverPolicy`, so it could mislead audits of the active hybrid lane. It now accepts
  `--bot` (default `solver_shop_basic_play_bot`) and traces the actual registry bot path.
- **Screened a plausible consumable double-count bug, but did not promote it:** trace scores
  showed very large `USE_CONSUMABLE` action values from leaf-delta terms. Two ablations failed
  the quality gate:
  - `BALATRO_USE_CONSUMABLE_LEAF_DELTA_W=0.0` with bonus-only scoring: **5/24 wins, mean ante
    6.04**, versus current first-24 **7/24, 6.25**.
  - `BALATRO_USE_CONSUMABLE_POTENTIAL_W=0.0`: tied the first-24 aggregate (**7/24, 6.25**) but
    churned seeds, then failed the 60-seed check: **15/60 wins, mean ante 6.50** versus current
    first-60 **16/60, 6.60**. Leave the current default unchanged.
- **Found and fixed stale Phase 8 shop-guidance wiring:** `basic_strategy.value_guidance`
  only knew about the old coarse `ml.features` / `phase8_value_model.npz` path, so a current
  `ValueNet` checkpoint could not drive the 1-step shop buy bonus. It now supports
  `BALATRO_VALUE_MODEL_CKPT=<checkpoint>` with `BALATRO_VALUE_MODEL_HEAD=ante|win|clear`, while
  preserving the old `BALATRO_VALUE_MODEL=1` npz path and keeping the feature off by default.
  A smoke with `.data/phase8_value_v3_bootstrap.pt` loaded successfully and returned a real
  delta; focused tests cover the new checkpoint path.

### 2026-06-04 -- seed-known portfolio diagnostic, not the target bot

- **Screened the fixed ValueNet shop-guidance hook as a bot decision signal:** using
  `.data/phase8_value_v3_bootstrap.pt`, `BALATRO_VALUE_GUIDED_HEAD=ante`, and low scales:
  - Basic + scale 25: **2/24 wins**, mean ante **5.79** vs Basic **2/24**, **5.62**; no win
    flips and a ~28% CPU tax before the cache/threading fix.
  - Basic + scale 50: **2/24 wins**, mean ante **5.88**; still no win flips.
  - Solver hybrid + scale 50: **5/24 wins**, mean ante **6.21** vs current solver **7/24**,
    **6.25**, with two solver wins lost and a large CPU tax. Do not promote current ValueNet
    1-step guidance as a quality default.
- **Reduced ValueNet guidance experiment overhead:** prediction now uses the BasicStrategy
  decision cache and single-thread batch-size-1 torch inference. An 8-seed timing sanity check
  dropped guided Basic from the earlier ~50s/run range to **~42.9s/run**, about baseline speed.
- **Added `portfolio_basic_solver_bot` as a diagnostic only:** it reads the seed string
  (`BALATRO_RUN_SEED` from the paired harness, or `modifiers["balatro_seed"]` from `SeedGame`)
  and simulates candidate full-run policies before choosing one to follow. This is a
  seed-known oracle-style selector; it is **not** counted as a meaningful general bot winrate
  and should not be treated as the target lane.
- **100-seed diagnostic result:** `portfolio_basic_solver_bot` vs current
  `solver_shop_basic_play_bot`, paired seeds `0000001..0000100`:
  **34/100 wins vs 22/100**, mean ante **7.09 vs 6.61**, paired d_ante **+0.48**, better/worse/same
  **26/0/74**, and **zero solver-win regressions**. The 12 win flips are exactly the Basic-only
  seeds from the prior overlap analysis: `0000010`, `0000015`, `0000026`, `0000030`, `0000037`,
  `0000039`, `0000059`, `0000067`, `0000070`, `0000081`, `0000089`, `0000090`. Metrics:
  `.data/bot_paired_solver_portfolio_basic_solver_100.json`.
- **Correct takeaway:** the useful information is not "34% winrate"; it is that Basic and the
  solver still win very different states. Future work should mine those Basic-only / solver-only
  divergences for state-local rules or learned action selectors that do **not** use full-seed
  rollout outcomes at deployment.

### 2026-06-04 -- rejected weak-joker Buffoon fallback

- **Tested a narrow non-seed-known rule from the Basic-only overlap:** ante 1, empty build,
  solver buys one of `Blue Joker` / `Greedy Joker` / `Seltzer`, and Basic opens a Buffoon
  pack. The rule was intentionally state-local and did not inspect the seed outcome.
- **Result:** first 24 seeds were a no-op (**7/24 wins, mean ante 6.25** on both bots). The
  offset 25-60 slice failed: **9/36 wins** on both bots, but mean ante fell
  **6.83 -> 6.78**, with better/worse/same **0/1/35** and zero win flips. The single changed
  seed was `0000039`, where the fallback made the solver drop from ante 5 to ante 3.
- **Cleanup:** removed the temporary `solver_shop_basic_play_buffoon_weak_joker_bot` alias and
  policy flag. This is a useful warning: copying Basic's first divergent shop move is not enough
  when the later play/shop policy context differs.
- **Validation stance:** seed-known portfolio numbers stay diagnostic only. General bot changes
  should pass contiguous seed slices and ideally separate discovery/holdout ranges before becoming
  defaults.

### 2026-06-04 -- holdout sanity check for seed fitting

- **Fresh contiguous holdout:** `basic_strategy_bot` vs current `solver_shop_basic_play_bot`,
  seeds `0000101..0000160`, `PYTHONHASHSEED=0`, same paired harness:
  Basic **10/60 wins**, mean ante **5.98**; hybrid **8/60 wins**, mean ante **6.33**.
  Paired better/worse/same **27/22/11**, hybrid win flips **8**, Basic win flips **10**.
  Metrics: `.data/bot_paired_basic_solver_shop_basicplay_holdout_101_160.json`.
- **Interpretation:** the current hybrid is not merely seed-fitting the first 100 seeds; it still
  improves depth on a fresh slice. But the first-100 **22%** winrate is not a stable expected
  arbitrary-seed winrate. On this holdout it loses the win-count comparison while retaining a
  survival/ante advantage. Treat current quality as "better depth, unstable win conversion" until
  larger held-out runs say otherwise.
- **Combined first-160 view:** current hybrid is **30/160 wins** vs Basic **23/160**, with mean
  ante **6.51 vs 5.93**, paired better/worse/same **81/48/31**, and win flips **29 vs 22**.
  That is evidence of a real general depth lift, but the winrate intervals still overlap; do not
  claim a stable arbitrary-seed winrate until a larger holdout confirms it.

### 2026-06-04 -- neural path reset: sim gate + candidate-ranker data

- **Stopped treating seed-known portfolio as the target:** pro-human white-stake play is an
  unseeded/general-policy objective. Portfolio remains only a diagnostic for Basic/solver
  complementarity.
- **Added a repeatable full-sim verification gate:** `scripts/full_sim_verification_gate.py`
  runs forward-sim tests, replay/score tests, RNG fixture validators, and score-edge fixtures
  with independent verifier tasks in parallel. Full no-live-bridge gate passed:
  `.data/full_sim_gate.json` (`forward_sim_tests` 147 tests + 19 subtests, replay/score 45 tests,
  RNG surface/shop/spectral validators, score fixtures).
- **Started the neural action-ranker path:** added `src/balatro_ai/ml/shop_candidate_dataset.py`
  and `scripts/phase8_shop_candidate_dataset.py`. The artifact is one JSONL row per shop state:
  encoded state (encoder v3), source bot, heuristic action, legal candidate actions, and
  common-random-number rollout values/ranks. This directly targets
  `score(state, candidate_action)` instead of the failed raw `V(state)` shop leaf.
- **Efficiency stance:** the sim gate runs independent verifier tasks with `--jobs`; candidate
  labeling uses `ProcessPoolExecutor` via `--jobs` and keeps smoke/default settings small enough
  for this machine. A one-record, two-worker smoke completed and wrote
  `.data/phase8_shop_candidates_smoke.jsonl` plus metrics.

### 2026-06-05 -- candidate-ranker label audit and first training probe

- **Found a real target bug before scaling data:** bounded rollout labels collapsed to flat
  ante values for candidates that all survived to the horizon. A 12-record probe had
  `mean_best_margin=0.0` and every best label was `buy card index 0`, which was just
  candidate-order tie-breaking.
- **Fixed the label value:** `rollout_value_after_action` now keeps ante/win survival as the
  dominant term, but adds a bounded same-horizon shop/build quality bonus from
  `shop_leaf_value`. This gives common-random-number labels enough resolution without letting
  the old heuristic override actual survival progress.
- **Fixed duplicate capture rows:** `scripts/phase8_shop_candidate_dataset.py` now over-collects
  cheap trajectory states, dedupes equivalent shop states across capture bots, and reports raw
  versus deduped counts.
- **Added the neural shop ranker and soft-label trainer:** `src/balatro_ai/ml/shop_ranker.py`
  scores `(state, candidate_action)` using the encoded state trunk plus action/shop-token
  features. The trainer supports hard argmax labels and soft rollout-value distributions via
  `--loss {soft,hard}` and `--target-temperature`.
- **Added an env-gated deployment wrapper:** `RankerGuidedShopBot` loads
  `BALATRO_SHOP_RANKER_CKPT` and scores shop candidates, with basic and
  solver-shop/basic-play registry aliases. No checkpoint means pure fallback.
- **Probe after fixes:** `.data/phase8_shop_candidates_probe_dedup.jsonl` captured 8 raw states,
  deduped to 4 unique rows, `mean_best_margin=0.0866`, and best actions were no longer a fixed
  first-card tie. Tiny training smoke with soft temperature 0.03 wrote
  `.data/phase8_shop_ranker_probe_dedup_soft_t003.metrics.json`: train top-1 1.0, all-row top-1
  0.75, all-row mean regret 0.0175. This is pipeline evidence only; the sample is far too small
  to claim playing strength.
- **Deployment smoke:** one seed with the tiny checkpoint loaded and ran through
  `solver_shop_basic_play_shop_ranker_bot`, but it regressed that seed from ante 8 to ante 3.
  Treat this as proof the wrapper executes, not as a usable model.
- **Verification:** focused ranker/dataset/registry tests pass, and the quick full-sim gate
  passed after the wrapper change (`.data/full_sim_gate_quick_after_ranker_wrapper.json`).

### 2026-06-05 -- candidate-data diversity and capture efficiency

- **Fixed first-shop/first-bot bias in candidate data:** `collect_shop_states` now accepts
  ante filters, and `scripts/phase8_shop_candidate_dataset.py` deterministically shuffles
  deduped states before slicing. This keeps larger datasets from being filled by the first
  capture bot's earliest shops.
- **Parallelized capture, not just labeling:** the dataset CLI now splits seed chunks across
  capture bots using `--collect-jobs` (defaulting to `--jobs`). This matters because later-shop
  capture itself was slow before any rollout labels were computed.
- **Added label-quality metrics:** metrics now include source/ante histograms, selected vs
  captured/deduped counts, split-half best-action agreement, nonzero best-margin rate, and
  mean top-tie count. These are needed because top-1 accuracy is arbitrary on tied rollout
  labels.
- **Ante-2 parallel smoke:** `.data/phase8_shop_candidates_ante2_parallel_4.jsonl` with
  `rollouts=2`, `max_actions=6`, `max_antes=1`, `jobs=8`, `collect_jobs=8` captured 32 states,
  deduped to 30, selected 4, and produced `mean_best_margin=0.1292`,
  `nonzero_best_margin_rate=0.75`, `mean_top_tie_count=2.25`, and
  `split_half_best_agreement_rate=0.5`. Training smoke wrote
  `.data/phase8_shop_ranker_ante2_parallel_4.metrics.json` with all-row top-1 0.75 and mean
  regret 0.0367. This is still a smoke; real data should use at least 4 CRN rollouts and judge
  held-out regret/ties, not raw top-1 alone.
- **Verification:** focused candidate/ranker/wrapper tests passed, and the quick full-sim gate
  passed after the parallel capture changes
  (`.data/full_sim_gate_quick_after_parallel_candidate_data.json`).

### 2026-06-05 -- soft/regret labels for multiple viable builds

- **User insight folded into the training target:** early shops often have several viable
  build basins on the same seed, so a single hard "best action" label is the wrong abstraction.
  The data and ranker metrics now report acceptable-action bands and near-best accuracy rather
  than relying on raw top-1.
- **4-rollout ante-2 dataset:** `.data/phase8_shop_candidates_ante2_r4_8.jsonl` captured
  64 states, deduped to 59, selected 8, balanced source bots 4/4, all ante 2. Label metrics:
  `mean_best_margin=0.0377`, `nonzero_best_margin_rate=0.625`, `mean_top_tie_count=1.875`,
  `mean_actions_within_0_05=2.25`, `mean_actions_within_0_10=3.0`,
  `heuristic_within_0_05_rate=0.6667`, and split-half agreement `0.375`.
- **Tiny ranker comparison:** on the same 6/2 split, soft mean-pool had the best held-out
  regret (`0.0721`) and near-best accuracy (`0.5`) among the tried variants. Hard labels
  overfit train top-1 but had much worse held-out regret (`0.2710`) and near-best accuracy
  `0.0`; attention-soft also underperformed on the tiny split. These are not playing-strength
  results, but they strongly support soft/regret labels for the next scale run.
- **Deployment decision:** no wrapper A/B was run from these checkpoints because the held-out
  sanity bar is not met. Next useful scale run should increase records first, keep 4+ CRN
  rollouts, and judge by held-out regret/near-best plus eventual paired bot A/B.
- **Verification:** focused candidate/ranker/wrapper tests passed and the quick full-sim gate
  passed after near-best metric changes (`.data/full_sim_gate_quick_after_near_best_metrics.json`).

### 2026-06-05 -- Rust best-play path fixed

- **Fixed the Rust best-play bridge instead of treating it as unsafe:** `rust_joker_data`
  now resolves Blueprint/Brainstorm like Python, with ability names/scaling metadata copied
  from the target joker while physical edition/rarity/sell value stay on the original slot.
  Copied Swashbuckler now uses copied metadata instead of incorrectly receiving the physical
  sell-value sum. The same Swashbuckler projection fix was applied to `state_value._cached_joker_data`.
- **Fixed Rust evaluator parity bugs found by real trajectory audits:** shape-gated jokers
  now inspect actual card rank counts; debuffed/Stone cards no longer count as faces for
  Photograph/Ride the Bus/Sock and Buskin-style checks; Ride the Bus uses that shared helper;
  and debuffed or suit-debuffed scoring cards skip the scored-card joker/effect pass, while
  still contributing to hand shape exactly like Python.
- **Added boss adjustment for the batch best-play path:** The Eye/The Mouth now post-process
  Rust score vectors using Python hand identification so repeated/disallowed hand types are
  zeroed before winner selection.
- **Removed the conservative best-play joker bailout list:** `RUST_BESTPLAY_UNSAFE_JOKERS`
  is now empty; unsupported cards/jokers still bail naturally through Rust returning `None`.
- **Verification:** rebuilt and force-reinstalled `balatro_core`. `cargo test` passes
  101 Rust tests; focused Python Rust suites pass (`tests/test_rust_bestplay_bridge.py`
  6 tests, `tests/test_rust_score_action_parity.py` 117 tests). Normal
  `BALATRO_BESTPLAY_PARITY=1 python scripts/bestplay_parity_check.py 4` now reports
  6,150 best-play calls, 5,692 Rust fast-path uses (92.6%), 458 bails (7.4%), and
  **0 vector divergences**.
- **Speed:** on `basic_strategy_bot` seed `0000001`, full-run wall time improved from
  25.5s with `BALATRO_RUST_BESTPLAY=0` to 8.27s with `BALATRO_RUST_BESTPLAY=1` on the
  same 163-step trajectory (~3.1x faster). The lane remains env-gated, but it is now a
  credible data-generation accelerator rather than a known-divergent experiment.

### 2026-06-05 -- Rust-backed candidate data smoke

- **Full sim gate re-run after Rust fixes:** `python scripts/full_sim_verification_gate.py
  --jobs 6 --metrics .data/full_sim_gate_after_rust_bestplay_fix.json` passed the full
  no-live-bridge gate in 1.35s across forward-sim, replay/score, RNG, fixture validators,
  and score-edge fixtures.
- **Candidate-data CLI now uses the safe Rust lane by default:** `scripts/phase8_shop_candidate_dataset.py`
  sets `BALATRO_RUST_BESTPLAY=1` before worker processes import the evaluator and records
  `rust_bestplay` in metrics. `--no-rust-bestplay` remains available for debugging/parity
  probes.
- **Fixed source-bot selection bias:** state selection can now round-robin across capture
  bot sources and the CLI defaults to that balanced mode. This avoids tiny datasets being
  accidentally dominated by one teacher trajectory distribution; metrics record
  `balance_source_bots`.
- **Multiworker smoke:** `.data/phase8_shop_candidates_ante2_r4_8_after_rust_balanced.jsonl`
  captured 64 states, deduped to 57, selected 8 balanced rows (4 basic / 4 solver-shop),
  `rollouts=4`, `max_antes=1`, `max_actions=8`, `jobs=8`, `collect_jobs=8`. Metrics:
  228 candidate continuations in 99.64s (2.29 continuations/s), `wall_s_per_record=12.46`,
  `mean_candidates=7.125`, `mean_best_margin=0.0485`, `mean_top_tie_count=2.625`,
  `mean_actions_within_0_05=3.125`, and split-half best agreement `0.625`.
- **Tiny train smoke:** soft mean-pool ranker on the balanced artifact wrote
  `.data/phase8_shop_ranker_ante2_r4_8_after_rust_balanced_soft.pt`. This is still a
  pipeline artifact only: all-row near-best@0.05 is 0.875, but the one-row held-out split
  fails top-1/near-best, so no deployment claim.

### 2026-06-05 overnight -- first 32-row balanced Rust-backed ranker checkpoint

- **Generated the first non-tiny balanced artifact:** `.data/phase8_shop_candidates_ante2_r4_32_after_rust_balanced.jsonl`
  with `states=32`, `rollouts=4`, `max_antes=1`, `max_actions=8`, `jobs=12`,
  `collect_jobs=12`, `seed_offset=920000`, source-balanced 16/16 between
  `basic_strategy_bot` and `solver_shop_basic_play_bot`. Metrics:
  980 candidate continuations in 324.22s (3.02 continuations/s),
  `wall_s_per_record=10.13`, `mean_best_margin=0.0534`, nonzero-margin rate
  `0.6875`, `mean_top_tie_count=1.9062`, `mean_actions_within_0_05=2.7812`,
  `mean_actions_within_0_10=3.5625`, and split-half best agreement `0.5625`.
- **Soft vs hard ranker sanity check:** both trained on the same seed split
  (`26 train / 6 val`, mean encoder, 120 epochs). Hard-label training memorized
  train top-1 (`1.0`) but failed held-out completely (`val top1=0.0`,
  `near_best_0_05=0.0`, regret `0.3101`). Soft labels were still weak but clearly
  better (`val top1=0.1667`, `near_best_0_05=0.3333`, regret `0.2052`).
  All-row metrics also favor soft on regret/near-best (`0.0385` regret and
  `0.875` near-best vs hard `0.0581` and `0.8125`).
- **Interpretation:** no deployment yet. The signal supports the current target design:
  train on soft/regret labels, scale records before adding architecture complexity, and
  judge by held-out regret/near-best rather than train top-1.

### 2026-06-05 resume-safe data generation checkpoint

- **Long candidate runs are now checkpointed:** `scripts/phase8_shop_candidate_dataset.py`
  writes ordered partial JSONL and partial metrics during the label pass by default
  (`--partial-every`, default 4). This avoids losing every completed shop-state label
  when a large multiworker run is interrupted.
- **Added resume support:** `--resume-partial` reloads records from the partial JSONL
  using `(source_bot, seed, state_index)` and labels only the remaining selected jobs.
  Final output remains ordered by the deterministic selected-state order, not by worker
  completion order.
- **Verification:** focused tests pass (`18 passed`). A 4-row multiworker smoke wrote
  final and `.partial` artifacts; rerunning the same command with `--resume-partial`
  reused all 4 records (`resumed_partial_records=4`, `remaining_label_jobs=0`) and
  finished in 11.59s instead of recomputing the label continuations.

### 2026-06-05 128-row candidate-ranker scale gate

- **Generated the first resume-safe 128-row artifact:** `.data/phase8_shop_candidates_ante2_r4_128_after_rust_balanced.jsonl`
  with `states=128`, `rollouts=4`, `max_antes=1`, `max_actions=8`, `jobs=12`,
  `collect_jobs=12`, `seed_offset=930000`, and `--resume-partial`. The run selected
  128 balanced ante-2 shop rows from 1,019 captured / 964 deduped states and finished
  cleanly with no stderr. Metrics: 3,828 candidate continuations in 948.22s
  (4.04 continuations/s), `wall_s_per_record=7.41`, source split 64/64,
  `mean_best_margin=0.0485`, nonzero-margin rate `0.5938`, split-half best agreement
  `0.6562`, and average near-best ambiguity of 2.875 actions within 0.05 and 3.523
  within 0.10.
- **Added missing heuristic baselines and best-val checkpoint selection:** ranker metrics
  now report heuristic train/val/all regret on the exact split, and training saves the
  epoch with best validation mean regret rather than the final epoch. Focused tests pass
  (`19 passed`).
- **128-row model verdict:** still not deployable. Best mean encoder selected epoch 110
  with held-out regret `0.1062`, near-best@0.05 `0.5833`; best attention selected epoch
  60 with held-out regret `0.1042`, near-best@0.05 `0.5833`. The heuristic baseline on
  the same val split is better: regret `0.0859`, near-best@0.05 `0.6957`. The all-row
  model numbers look good because the models fit train; held-out says scale data and/or
  improve labels before downstream A/B.

### 2026-06-05 label-quality and horizon-2 probe

- **Added train-label quality controls:** ranker examples now retain best-margin,
  split-half agreement, and near-best action counts from the rollout labels. The trainer
  can filter the training split by these fields and reports quality summaries for all,
  train, filtered-train, and val slices. This made the noisy-label hypothesis testable
  without changing the held-out validation slice.
- **Filtering the 128-row horizon-1 data did not help:** margin-only and stable/low-ambiguity
  filters reduced the training set to 47 and 27 rows respectively and both worsened held-out
  regret versus the unfiltered model and the heuristic. Conclusion: do not solve the
  horizon-1 failure by throwing away ambiguous rows; it becomes data-starved.
- **Added repeated split-sweep tooling:** `scripts/phase8_ranker_split_sweep.py` trains
  across multiple seed splits and compares neural regret/near-best against the stored
  heuristic action, avoiding one lucky validation split.
- **Horizon-2 labels are slower but more stable/tied:** `.data/phase8_shop_candidates_ante2_r4_m2_32_after_rust_balanced.jsonl`
  used `max_antes=2`, `states=32`, `rollouts=4`, `jobs=12`, and source balancing. It
  finished in 596.14s: 916 continuations at 1.54/s, 16/16 source split, split-half best
  agreement `0.9375`, but low `mean_best_margin=0.0190` with 4.06 actions within 0.05.
- **Tiny horizon-2 attention sweep is promising but not deployable:** repeated seed-split
  sweep over 7 split seeds (`.data/phase8_ranker_sweep_ante2_r4_m2_32_attention.metrics.json`)
  has attention mean regret `0.0503` vs heuristic `0.1066`, winning regret on all 7 splits;
  near-best@0.05 is `0.7517` vs heuristic `0.6000`, winning 5/7. The same sweep on the
  128-row horizon-1 artifact (`.data/phase8_ranker_sweep_ante2_r4_m1_128_attention.metrics.json`)
  still loses on average: model regret `0.1079` vs heuristic `0.0969`, near-best@0.05
  `0.5460` vs `0.6201`. Next gate: scale horizon-2 labels before any bot deployment.

### 2026-06-05 64-row horizon-2 scale confirmation

- **Scaled the promising horizon-2 label path to 64 rows:** `.data/phase8_shop_candidates_ante2_r4_m2_64_after_rust_balanced.jsonl`
  used `states=64`, `rollouts=4`, `max_antes=2`, `max_steps=420`, `max_actions=8`,
  `jobs=12`, source balancing, Rust best-play, and resume checkpoints. It finished cleanly
  with no stderr: 634 captured -> 601 deduped -> 64 selected, 1,928 candidate
  continuations in 1,237.53s (1.56/s), 32/32 source split.
- **Label diagnostics stayed horizon-2-shaped:** `mean_best_margin=0.0297`, nonzero-margin
  rate `0.375`, `mean_top_tie_count=3.1562`, `mean_actions_within_0_05=3.8906`,
  `mean_actions_within_0_10=4.6406`, split-half best agreement `0.8281`.
- **Attention ranker repeated split sweep confirms the direction:** `.data/phase8_ranker_sweep_ante2_r4_m2_64_attention.metrics.json`
  over the standard 7 seed splits reports mean model regret `0.0729` vs heuristic
  `0.1217`, with regret wins 7/7. Near-best@0.05 is `0.6576` vs heuristic `0.5847`,
  winning 4/7 splits. This is weaker than the tiny 32-row probe on near-best but much
  stronger evidence than horizon-1 because it survives more rows and repeated splits.
- **Current verdict:** still no deployment claim, because this is offline candidate ranking
  on ante-2 shop states only. But horizon-2 attention is now the best neural signal found:
  scale it next (128 rows or mixed ante-2/3) and then test a ranker-guided shop bot only
  after repeated split metrics stay ahead of heuristic.

### 2026-06-05 combined horizon-2 offline/online gate

- **Combined existing horizon-2 datasets instead of immediately spending more rollout CPU:**
  `scripts/phase8_train_shop_ranker.py` and `scripts/phase8_ranker_split_sweep.py` now accept
  repeated `--data` arguments and dedupe examples by `(source_bot, seed, state_index)`.
  The 32-row and 64-row horizon-2 artifacts combine into a 96-row offline gate.
- **Matched training action space to safe deployment:** ranker loading can filter candidate
  action types before recomputing the best label. For live testing, the shop-ranker wrapper
  also supports `BALATRO_SHOP_RANKER_ACTION_TYPES` and
  `BALATRO_SHOP_RANKER_MAX_ACTIONS_PER_SHOP`. Focused tests pass (`77 passed`) after these
  changes.
- **Combined 96-row horizon-2 attention sweep remains strong offline:**
  `.data/phase8_ranker_sweep_ante2_r4_m2_96_attention.metrics.json` reports mean regret
  `0.0535` vs heuristic `0.0960`, regret wins 7/7, near-best@0.05 `0.7458` vs `0.6071`,
  near-best wins 7/7. The action-space-matched safe-action sweep
  `.data/phase8_ranker_sweep_ante2_r4_m2_96_attention_safeactions.metrics.json` is also
  strong: mean regret `0.0463` vs heuristic `0.0756`, regret wins 7/7, near-best@0.05
  `0.7899` vs `0.6524`, near-best wins 6/7.
- **Downstream wrapper smoke exposed the offline/online gap:** unconstrained live ranker was
  catastrophic, because it could chain SELL/REROLL actions and sell key jokers. Gating to
  `buy,open_pack,end_shop` fixed the obvious failure mode. On a deterministic 24-seed paired
  smoke (`PYTHONHASHSEED=0`, ante-2 only, no sell/reroll,
  `.data/bot_paired_solver_shop_ranker_h2_96_ante2_safeactions_pyhash0_24.json`), the ranker
  wrapper is only slightly positive: 6 wins vs 5 for baseline, mean ante +0.083, better 9 /
  worse 10 / same 5. This is not promotion-worthy yet.
- **Current verdict:** horizon-2 attention is a real offline signal, and safe action gating
  prevents catastrophic live behavior, but the online edge is too small/noisy. Next gate:
  scale horizon-2 labels to at least 128 combined rows or add ante-3 horizon-2 data, then run
  a deterministic 48+ seed safe-action A/B only if the repeated split sweep stays ahead.

### 2026-06-05 overnight ante-balance checkpoint

- **Added opt-in ante-balanced state selection:** `scripts/phase8_shop_candidate_dataset.py`
  now supports `--balance-antes`, which round-robins selected states across `(source_bot, ante)`
  groups when used with the default source balancing. Focused tests pass (`78 passed`).
- **Mixed ante-2/3 collection attempt completed but exposed a collector quota issue:**
  `.data/phase8_shop_candidates_ante2to3_r4_m2_64_after_rust_balanced.jsonl` finished
  cleanly with no stderr: 768 captured -> 714 deduped -> 64 selected, 1,864 candidate
  continuations in 1,293.63s (1.44/s), 32/32 source split. However `records_by_ante` is
  `{"2": 64}` despite `--max-capture-ante 3` and `--balance-antes`.
- **Do not treat this as ante-3 coverage:** the likely cause is the collector filling each
  seed's `--per-seed 2` quota with the first two ante-2 shops before ante-3 states enter the
  candidate pool. Keep this artifact as extra ante-2 horizon-2 data, but do not train a
  "mixed ante" model from it. Next fix: make collection ante-balanced too, increase
  `--per-seed`, or run an explicit ante-3-only capture before the next sweep.

### 2026-06-05 mixed-ante v2 labels and online gate

- **Fixed mixed-ante collection and label semantics:** `--balance-antes` now applies during
  collection as a per-ante quota, not just final selection. Same-horizon terminal labels now
  include a small resource/economy floor bonus (`label_value_version=2`), so a richer/scalier
  line can beat a slightly higher immediate-clear line when both survive to the same horizon.
  Focused tests pass (`85 passed` after wrapper fixes).
- **Made safe-action data generation cheaper and better matched to deployment:**
  `scripts/phase8_shop_candidate_dataset.py --candidate-action-types buy,open_pack,end_shop`
  filters candidates before expensive rollouts. The safe-action 64-row mixed ante artifact
  `.data/phase8_shop_candidates_ante2to3_r4_m2_64_resourcefloor_safeactions.jsonl` finished
  cleanly: 768 captured -> 737 deduped -> 64 selected, exact 32/32 source split, exact 32/32
  ante split, all label v2, 1,348 continuations in 1,227.81s. This is substantially cheaper
  than the all-action corrected artifact (1,976 continuations in 1,744.76s).
- **Offline ranker signal is real but still small-data:** safe-action mean encoder sweep
  `.data/phase8_ranker_sweep_ante2to3_r4_m2_64_resourcefloor_safeactions_mean_attention.metrics.json`
  reports mean regret `0.0599` vs heuristic `0.1168`, regret wins 5/7, near-best@0.05
  `0.8132` vs heuristic `0.7560`. Attention is worse here (`0.0751` regret). Full-action
  corrected labels are only mildly ahead offline (attention regret `0.0786` vs heuristic
  `0.0940`, 4/7 wins), so they are not the next deployment lane.
- **Online replacement still fails:** trained
  `.data/phase8_shop_ranker_ante2to3_r4_m2_64_resourcefloor_safeactions_mean.pt`. Deterministic
  12-seed paired smokes against `solver_shop_basic_play_bot` are negative: no cap was 1/12 wins
  vs 4/12, mean ante -1.333; cap=1 was 0/12 before fixing a booster-session cap leak and 1/12
  after; cap=1 plus `BALATRO_SHOP_RANKER_MIN_MARGIN=0.5` improved to 2/12, mean ante -0.417;
  margin 1.0 nearly neutralized but still lost, 3/12 vs 4/12, mean ante -0.167. Added a
  baseline-comparison gate (`BALATRO_SHOP_RANKER_COMPARE_BASELINE=1`) that probes the wrapped
  solver on a deep-copied bot and only allows neural overrides over covered safe candidates;
  it is mechanically safer but still negative on the same 12-seed lane:
  `.data/bot_paired_solver_shop_ranker_ante2to3_safeactions_resourcefloor_mean_cap1_margin10_comparebaseline_12.json`
  has ranker 2/12 wins vs baseline 4/12, mean ante -0.25, better 2 / worse 4 / same 6, and
  higher mean CPU (47.82s vs 27.52s).
- **Diagnosis:** the ranker can reduce offline regret, but free-running replacement is the
  wrong deployment shape. Safe-action training cannot compare against the solver's sell/reroll
  choices, and one-action labels do not authorize repeated ranker control across a shop. The
  baseline-comparison wrapper confirms that gating a one-action ranker is not enough. Next
  neural gate should be an override/advantage model: label the baseline action/continuation
  alongside alternatives, predict advantage over that baseline, and only override when predicted
  advantage is large and validated on paired online seeds.
- **Added baseline-aware advantage training/eval:** shop-ranker examples now carry
  `baseline_index`, `baseline_value`, and candidate-minus-baseline `advantages`; batches carry
  advantage tensors; training supports `--loss advantage_mse`; and train/sweep metrics now
  report thresholded override lift/harm over the current solver action. Focused tests pass
  (`87 passed`). On the existing 64-row mixed ante safe-action artifact, repeated mean-encoder
  split sweep with `advantage_mse`
  (`.data/phase8_ranker_sweep_ante2to3_r4_m2_64_resourcefloor_safeactions_mean_advantage_mse.metrics.json`)
  is aligned offline: mean lift vs baseline `+0.0906`, regret delta `-0.0906`, positive in
  6/7 splits, but with high override rate (`0.971`) and nonzero harm (`0.0567`).
- **Advantage checkpoint still fails online unless made inert:** trained
  `.data/phase8_shop_ranker_ante2to3_r4_m2_64_resourcefloor_safeactions_mean_advantage_mse.pt`.
  With baseline comparison, cap=1, safe actions, and baseline margin `0.10`, the 12-seed paired
  lane is still negative:
  `.data/bot_paired_solver_shop_ranker_ante2to3_safeactions_resourcefloor_mean_advantage_mse_baseline_margin010_12.json`
  has ranker 2/12 wins vs baseline 4/12, mean ante -0.417, better 4 / worse 5 / same 3. Raising
  the baseline margin to `0.30`
  (`.data/bot_paired_solver_shop_ranker_ante2to3_safeactions_resourcefloor_mean_advantage_mse_baseline_margin030_12.json`)
  makes the run exactly neutral (4/12 vs 4/12, all 12 same), meaning the threshold only becomes
  safe when it suppresses essentially all neural overrides.
- **Trace diagnosis:** on baseline-win regression seed `0300005`, the neural run dies at ante 3
  while baseline wins. The first damaging divergence is an ante-2 shop where the model opens a
  Jumbo Celestial pack over the solver's `end_shop` with predicted baseline margin `0.27193`.
  That detour changes the future shop/economy path and loses the Card Sharp/Half Joker/Ramen/Bull
  line the solver later finds. This confirms the current horizon-2 labels still over-credit some
  pack/buy detours; the next dataset should use deeper/full-run baseline-vs-candidate advantage
  labels, with particular attention to pack openings and money preservation.
- **Fixed the baseline coverage mismatch for advantage labels:** candidate records can now
  `--include-heuristic-action`, which appends the solver/heuristic action as a comparison-only
  candidate even when `--candidate-action-types buy,open_pack,end_shop` filters it out. Ranker
  loading/training can now use `--keep-heuristic-action` so safe neural candidates are compared
  against unsafe solver baselines such as reroll/sell without allowing the neural policy to return
  those unsafe actions. The live wrapper mirrors this: it may score the wrapped solver action as
  baseline, but chooses overrides only from `BALATRO_SHOP_RANKER_ACTION_TYPES`. Focused tests pass
  (`90 passed`).
- **Deep advantage smoke:** generated
  `.data/phase8_shop_candidates_deep_advantage_includeheuristic_smoke.jsonl` with safe candidates
  plus retained solver action, `max_antes=8`, 2 CRN rollouts, balanced ante/source selection.
  It completed 4 rows / 50 candidate continuations in 370.57s (`0.135` continuations/s), with
  exact 2/2 source and 2/2 ante split, `heuristic_present_rate=1.0`, `heuristic_best_rate=0.25`,
  mean best margin `1.0851`, and one retained baseline outside the safe action set (`reroll`).
  The tiny train CLI smoke
  `.data/phase8_shop_ranker_deep_advantage_includeheuristic_smoke.metrics.json` verifies
  `--loss advantage_mse --keep-heuristic-action` end-to-end with 100% baseline coverage. This is
  not a trainable artifact; it proves the next scale lane and exposes the cost.
- **Added focused candidate budgeting for deep labels:** `candidate_shop_actions(...,
  priority="deep_advantage")` now orders candidates as `end_shop`, pack opens, buys, rerolls,
  sells before `max_actions` truncation. The dataset CLI exposes this as
  `--candidate-priority deep_advantage`. A comparable focused smoke
  `.data/phase8_shop_candidates_deep_advantage_focused_smoke.jsonl` used `--max-actions 4`
  and completed the same 4 selected states with 38 candidate continuations in 283.05s, reducing
  continuation count and wall time by about 24% while preserving exact source/ante balance and
  `heuristic_present_rate=1.0`. It retained one comparison-only `reroll` baseline outside the
  safe neural action set. The tiny train smoke
  `.data/phase8_shop_ranker_deep_advantage_focused_smoke.metrics.json` confirms
  `--loss advantage_mse --keep-heuristic-action` still runs end-to-end on the focused artifact.
  Focused tests pass (`91 passed`).
- **First modest focused deep artifact:** added dataset metrics for `heuristic_action_types` and
  comparison-only baselines outside the safe candidate action types. Generated
  `.data/phase8_shop_candidates_deep_advantage_focused_8.jsonl` with 8 rows, `max_antes=8`,
  2 rollouts, `--candidate-priority deep_advantage --max-actions 4`, 8 workers, and partial
  checkpoints. It completed 70 candidate continuations in 327.31s (`0.214/s`, 40.9s/record),
  exact 4/4 source split and 4/4 ante split, `heuristic_present_rate=1.0`, heuristic action
  types `buy=7, reroll=1`, comparison-only outside-safe rate `0.125`, mean best margin `0.8967`,
  and split-half agreement `0.5`. The 8-row train artifact
  `.data/phase8_shop_ranker_deep_advantage_focused_8.metrics.json` is still too small to deploy
  but verifies the scaled artifact path.
- **Tiny repeated split check on 12 focused deep rows:** combining the 4-row focused smoke and
  the 8-row focused artifact in
  `.data/phase8_ranker_sweep_deep_advantage_focused_12_mean_advantage_mse.metrics.json`
  gives a noisy but encouraging advantage signal: mean model regret `1.0953` vs heuristic
  `1.8602`, model regret wins 7/7, mean advantage lift vs baseline `+0.8644`, advantage regret
  delta `-0.8644`, positive lift 7/7. Harmful override rate remains nonzero (`0.1429`), so this
  is not deployable; it supports scaling the same deep-focused lane before another online A/B.
- **Focused deep 16/28 check and economy label correction:** generated
  `.data/phase8_shop_candidates_deep_advantage_focused_16.jsonl` with 16 rows, `max_antes=8`,
  2 rollouts, exact 8/8 source and ante splits, 150 candidate continuations in 929.50s, and
  `heuristic_present_rate=1.0`. It finally includes meaningful comparison-only solver baselines
  outside the safe neural action set (`sell=5`, `reroll=2`, outside-safe rate `0.4375`). The
  28-row repeated split sweep
  `.data/phase8_ranker_sweep_deep_advantage_focused_28_mean_advantage_mse.metrics.json` improves
  average regret on every split (`0.8469` model vs `1.4799` heuristic; mean advantage lift
  `+0.4281`), but the override policy is still too eager (`0.3905` harmful override rate), so it
  is not deployable. Also audited the label semantics against the user's economy concern: v2 made
  resource/economy mostly a floor. `LABEL_VALUE_VERSION=3` now adds an explicit bounded resource
  bonus for same-horizon survivors, so a safely clearing line with a healthier bankroll remains
  distinguishable from a slightly higher immediate/build score. Focused tests pass (`92 passed`).
  Existing v2 artifacts are diagnostic only; regenerate v3 labels before the next scale or online
  A/B.
- **V3 focused deep 32-row gate:** generated two independent v3 focused deep shards:
  `.data/phase8_shop_candidates_deep_advantage_focused_v3_16.jsonl` and
  `.data/phase8_shop_candidates_deep_advantage_focused_v3_16b.jsonl`. Both are exact 8/8 source
  and ante splits with retained solver baselines. Shard A completed 150 continuations in 673.19s,
  split-half agreement `0.625`, outside-safe baseline rate `0.4375`; shard B completed 146
  continuations in 794.95s, split-half agreement `0.3125`, outside-safe baseline rate `0.1875`.
  Combined 32-row quality: mean best margin `0.7035`, split-half agreement `0.46875`.
  The combined attention ranker is the best v3 signal so far:
  `.data/phase8_ranker_sweep_deep_advantage_focused_v3_32_attention_advantage_mse.metrics.json`
  reports model regret `0.7233` vs heuristic `0.9500`, near-best@0.05 `0.4741` vs `0.3367`,
  and top-1 `0.4095` vs `0.2721`. But override calibration is still unsafe: at threshold
  `0.1`, mean lift is only `+0.1559`, positive in 2/7 splits, with harmful override rate
  `0.3476`; threshold `0.3` reduces harm to `0.2143` but is positive in only 1/7 splits;
  threshold `0.5` still has harm `0.2857`. Training only on split-half-agreeing rows is worse
  and starves the model. Verdict: do not run online A/B or promote a checkpoint. The ranker is
  learning useful ordering, but advantage gating needs cleaner/more stable labels.
- **Rollouts=4 probe and snapshot relabel path:** generated
  `.data/phase8_shop_candidates_deep_advantage_focused_v3_r4_8.jsonl` as a small v3 focused
  `rollouts=4` label-quality probe. It was not better: 8 rows took 144 continuations in
  866.75s, split-half agreement fell to `0.125`, and tiny repeated sweeps lost to the heuristic
  (attention regret `0.8783` vs heuristic `0.5139`, harmful override rate `0.6429`; mean regret
  `0.9377` vs heuristic `0.5139`). This argues against blindly scaling fresh `r4` collection.
  To make future label-quality work cheaper, candidate records now include a reloadable
  `state_snapshot`, and `scripts/phase8_shop_candidate_dataset.py --input-records` can relabel
  those exact selected states with new rollout counts/settings without recollecting trajectories.
  Smoke artifacts `.data/phase8_snapshot_relabel_smoke_source.jsonl` and
  `.data/phase8_snapshot_relabel_smoke_relabel.jsonl` verify the path end-to-end: the second run
  consumed input records, doubled continuations from 12 to 24, and avoided a fresh capture pass.
  Focused tests pass (`39 passed`). Existing older candidate JSONL files do not have snapshots;
  regenerate once with the current CLI before using `--input-records`.
- **Same-state r2/r4 relabel comparison:** generated a fresh snapshot-bearing 8-state shard
  `.data/phase8_same_state_v3_r2_8.jsonl` (`rollouts=2`, 70 continuations in 298.17s,
  split-half agreement `0.375`) and relabeled the exact same states through `--input-records`
  as `.data/phase8_same_state_v3_r4_8.jsonl` (`rollouts=4`, 140 continuations in 592.01s,
  split-half agreement still `0.375`). The r4 attention sweep is better on this tiny state set
  (regret `0.2772` vs heuristic `0.7518`, harmful override rate `0.0`) while r2 has regret
  `0.4676` vs heuristic `1.1365`, harmful override `0.1429`. However, the direct label diff is
  the important signal: the best action changed on 5/8 states (`0.625`), with mean absolute
  shared-candidate value delta `0.9403` and max delta `2.2970`. More CRN seeds materially change
  labels but do not solve split-half instability. Treat r2 deep labels as too noisy for
  deployment calibration; r4 is more plausible but needs a larger same-state/snapshot gate before
  any online test.
- **Same-state r4/r8 label-noise gate:** generated a reusable balanced 16-state snapshot pool
  `.data/phase8_capture_pool_v3_16.jsonl` and relabeled it as
  `.data/phase8_capture_pool_v3_r4_16.jsonl` (`rollouts=4`, 292 candidate continuations,
  1383.48s, 8 workers). The r4 pool has exact 8/8 source and ante splits, but split-half
  agreement is only `0.25`. Repeated split sweeps do not justify deployment: attention is
  slightly worse than the heuristic on raw regret (`0.6071` vs `0.5876`), while mean is slightly
  better (`0.5253` vs `0.5876`) but unsafe as an override model. The cleanest-looking attention
  gate at threshold `0.3` has mean lift `+0.0836` and harmful override rate `0.0714`, but it is
  positive in only 2/7 splits and overrides only `0.131` of covered examples. This is too small
  and too fragile for online A/B.
- **Higher-rollout same-state r8 check:** relabeled the exact same 16-state snapshot pool as
  `.data/phase8_capture_pool_v3_r8_16.jsonl` (`rollouts=8`, 584 continuations, 2858.25s /
  47.6 minutes, 8 workers). Doubling rollouts did not improve the core stability metric:
  split-half agreement stayed `0.25`, mean best margin fell to `0.3118`, and the heuristic was
  within `0.05` of the labeled best on half the states. Compared with r4, best action changed on
  3/16 exact states (`0.1875`), but shared-candidate values still moved substantially
  (mean absolute delta `0.4358`, max `1.4950`). R8 ranker sweeps also do not clear the gate:
  attention regret `0.5356` vs heuristic `0.4100`, mean regret `0.4123` vs heuristic `0.4100`,
  and thresholded overrides remain negative or harmful on average. Training only on
  `best_margin >= 0.25` examples did not fix calibration and starved each split to roughly
  6-7 training rows. Conclusion: more CRN seeds alone is not enough. Early/deep shop states often
  have several viable branches, so a single "best action" argmax target creates fake precision.
- **Cheap breadth lane established:** capture-only state pools are now clearly worthwhile. A
  larger reusable pool, `.data/phase8_capture_pool_v3_64.jsonl`, captured 64 balanced ante-2/3
  shop snapshots in 172.06s with exact 32/32 source and ante splits. This decouples cheap state
  collection from expensive relabeling and should be the basis for the next label-design
  experiments instead of repeatedly recollecting trajectories.
- **Confidence-aware label audit added:** added paired-CRN rollout confidence diagnostics in
  `balatro_ai.ml.shop_candidate_dataset`, a standalone
  `scripts/phase8_shop_confidence_audit.py`, and ranker example/filter fields for best-vs-runner-up
  and best-vs-heuristic lower confidence bounds. Focused tests pass (`44 passed`). Auditing the
  r8 16-state pool shows why one-best labels are failing: only `12.5%` of sampled
  best-vs-runner-up choices have a positive lower bound, while `87.5%` are ambiguous
  (`.data/phase8_capture_pool_v3_r8_16.confidence.json`). Best-vs-heuristic has more actionability
  but is still sparse: only `25%` of states have a practical high-confidence override candidate.
  R4 is similar (`18.75%` best-vs-runner-up high confidence; `81.25%` ambiguous). A confidence
  filtered mean/`advantage_tie_mse` sweep that trained only on examples with
  `best_vs_baseline_lcb >= 0.05` starved to 2-3 fit examples per split and still lost badly:
  regret `0.6439` vs heuristic `0.4561`, mean lift `-0.1878`, harmful override `0.5833`
  at threshold `0.0`
  (`.data/phase8_ranker_sweep_capture_pool_v3_r8_16_mean_advantage_tie_m010_conf_baseline_lcb005.metrics.json`).
  Conclusion: the right target is confidence-aware, but the current tiny pool does not contain
  enough high-confidence baseline-vs-candidate examples to train a deployable neural shop model.
- **Targeted state selection for efficient relabeling:** added
  `scripts/phase8_select_shop_state_pool.py` to cheaply select snapshot records before expensive
  rollout labeling. On the 64-state capture pool, the solver heuristic is `sell`/`reroll`/`end_shop`
  on 27/64 states and outside the focused `buy/open_pack/end_shop` candidate set on 41/64, so
  targeted selection is more sensible than random relabeling. The first targeted 16-state pool
  (`.data/phase8_capture_pool_v3_64_targeted_16.jsonl`) is perfectly source/ante balanced and has
  16/16 solver baselines outside the focused candidates, but it over-focuses on `sell` baselines.
  A more diverse pool (`.data/phase8_capture_pool_v3_64_targeted_diverse_16.jsonl`) remains
  source-balanced and spans heuristic actions (`buy=4`, `end_shop=4`, `open_pack=2`, `reroll=3`,
  `sell=3`). A cheap 4-state r2/short-horizon probe of the sell-heavy pool found no confident
  override signal. The diverse 4-state r2/short probe is better: best-vs-runner-up is still fully
  ambiguous, but best-vs-heuristic has a `25%` practical high-confidence rate, driven by an ante-3
  `end_shop` baseline where a pack-open line has a large positive paired lower bound. This suggests
  the next labels should balance heuristic action type and then deepen only baseline-vs-candidate
  promising states.
- **Adaptive deepening selector and r2 false-positive check:** added
  `scripts/phase8_select_deepening_states.py`, which reads shallow candidate labels and emits a
  small state-snapshot pool ranked by paired candidate-minus-heuristic confidence. It records and
  can require rollout count, so r2 probes can be treated as exploratory instead of accepted as
  high-confidence data. Applied to the diverse r2/short smoke, it selected the one promising
  ante-3 `end_shop` vs `open_pack` state
  (`.data/phase8_targeted_diverse16_deepen_from_r2_smoke.jsonl`, LCB `1.7621`, only 2 rollouts).
  The same gate with `--min-rollouts 4` correctly rejects the r2-only evidence. Deepening that
  single state to r4/max_antes=8 took 231.47s for 16 continuations
  (`.data/phase8_targeted_diverse_deepen1_r4_m8.jsonl`) and disproved the r2 confidence: mean
  best-vs-heuristic advantage stayed positive (`0.7564`), but SEM was `1.5167` and the lower bound
  fell to `-0.7602`, so the adaptive selector rejects it at `min_rollouts=4`. Conclusion: r2/short
  can discover interesting candidates, but the training gate must require same-horizon r4+ paired
  confidence or use sequential sampling until the lower bound stabilizes.
- **Sequential baseline-vs-candidate probe:** added
  `scripts/phase8_sequential_baseline_probe.py`, a multiworker paired baseline probe that samples
  candidate-minus-heuristic rollouts sequentially and stops candidates once their paired LCB/UCB is
  clearly positive or negative. It writes compatible candidate JSONL plus `sequential_*` audit
  fields and has a per-state wall-clock budget to prevent one bad state from consuming the whole
  run. Focused tests pass (`52 passed`). A first deep smoke
  (`states=2`, `min_rollouts=4`, `max_rollouts=8`, `max_antes=8`) was stopped after exceeding the
  15-minute tool timeout, confirming that the current deep continuation cost is still too high for
  casual scale. A safer shallow smoke
  `.data/phase8_sequential_baseline_probe_diverse2_r2to4_m4.jsonl` completed 2 states in 130.77s
  with 2 workers and the 120s per-state budget; it returned partial 2-rollout records with
  `state_timeout` stop reasons and no high-confidence overrides. Conclusion: sequential probing is
  the right safety/efficiency wrapper, but the next blocker is rollout continuation speed/cost,
  not model architecture.
- **Rollout cost profile and two-stage confirmation lane:** added
  `scripts/phase8_rollout_cost_profile.py`. On targeted diverse state 0/action 0/seed 1,
  `solver_shop_basic_play_bot` spent `18.42s` per continuation with `99.1%` of wall time inside
  `choose_action`; `basic_strategy_bot` spent `1.16s` on the same continuation and reached the same
  terminal value. This points to a two-stage data lane: fast basic rollouts for exploration, then
  focused solver confirmation. A 4-state basic-rollout sequential exploration
  (`.data/phase8_sequential_baseline_probe_diverse4_basic_r4to8_m8.jsonl`) found practical
  high-confidence best-vs-heuristic candidates on 3/4 states in 192.92s. Filtering with
  `phase8_select_deepening_states.py --min-rollouts 4` produced one robust exploration candidate:
  ante-2 `end_shop -> open_pack`, LCB `0.177`
  (`.data/phase8_basic_explore_diverse4_deepen_candidates_minr4.jsonl`). Added
  `--focus-deepening-candidate` to the sequential probe so confirmation samples only the chosen
  candidate plus the heuristic. Focused solver confirmation of that one candidate used 8
  continuations in 248.02s and kept a positive r4/max_antes=8 paired lower bound (`+0.320`);
  the unfocused confirmation had used 13 continuations in 353.89s. This is the first clean
  evidence for the efficient label recipe: cheap teacher explores, adaptive filter selects, strong
  solver confirms one candidate before anything reaches training.
- **Diverse-16 two-stage mini-funnel:** scaled the efficient lane from 4 exploratory states to
  the full diverse 16-state pool. Basic-rollout exploration
  (`.data/phase8_sequential_baseline_probe_diverse16_basic_r4to8_m8.jsonl`) ran 16 states with
  8 workers in 391.82s, producing 245 candidate continuations and high-confidence practical
  best-vs-heuristic signals on 8/16 states. Requiring at least 4 paired samples through
  `phase8_select_deepening_states.py` narrowed this to two solver-confirmation candidates:
  ante-2 `buy -> open_pack` and ante-3 `buy -> end_shop`
  (`.data/phase8_basic_explore_diverse16_deepen_candidates_minr4.jsonl`). Focused solver
  confirmation of both candidates
  (`.data/phase8_solver_confirm_basic_explore_diverse16_top2_focused_r4_m8.jsonl`) finished in
  206.73s with 2 workers and 16 total continuations. One candidate survived as a real
  solver-confirmed r4/max_antes=8 positive label: ante-2 `buy -> open_pack`, mean advantage
  `+1.425`, SEM `0.781`, LCB `+0.645`, positive sample rate `0.75`
  (`.data/phase8_solver_confirmed_positive_labels_diverse16_top2_minr4.jsonl`). The ante-3
  `buy -> end_shop` candidate stayed mean-positive but ambiguous (LCB `-0.843`). This funnel is
  sparse but working: fast exploration found candidates cheaply, and focused solver confirmation
  filtered false positives before training.
- **Second non-overlapping diverse-16 funnel:** added `--exclude-records` to
  `scripts/phase8_select_shop_state_pool.py` and selected
  `.data/phase8_capture_pool_v3_64_targeted_diverse_16b.jsonl`, excluding the first diverse pool.
  The second pool uses 48 remaining records, selects 16 states, and spans `buy=5`, `end_shop=4`,
  `reroll=3`, `sell=4` heuristic actions. Basic-rollout exploration
  (`.data/phase8_sequential_baseline_probe_diverse16b_basic_r4to8_m8.jsonl`) ran in 390.93s with
  8 workers, producing 246 candidate continuations and practical high-confidence best-vs-heuristic
  signals on 5/16 states. The r4/min-rollout adaptive filter selected 4 solver-confirmation
  candidates (`.data/phase8_basic_explore_diverse16b_deepen_candidates_minr4.jsonl`). Focused
  solver confirmation
  (`.data/phase8_solver_confirm_basic_explore_diverse16b_top4_focused_r4_m8.jsonl`) completed all
  4 in 258.77s with 4 workers. One candidate survived: ante-2 `end_shop -> buy`, mean advantage
  `+2.151`, SEM `1.145`, LCB `+1.006`, positive sample rate `0.75`
  (`.data/phase8_solver_confirmed_positive_labels_diverse16b_top4_minr4.jsonl`). Two candidates
  were solver-rejected via negative UCB, and one was mean-positive but ambiguous. Combined
  two-pool funnel so far: 32 fast-explored states -> 6 solver-confirmed candidates -> 2
  solver-confirmed positive labels.
- **Third non-overlapping diverse-16 funnel:** selected
  `.data/phase8_capture_pool_v3_64_targeted_diverse_16c.jsonl`, excluding both prior diverse
  pools. The third pool uses the remaining 32 records, selects 16 states, and spans `buy=8`,
  `end_shop=1`, `reroll=2`, `sell=5` heuristic actions. Basic-rollout exploration
  (`.data/phase8_sequential_baseline_probe_diverse16c_basic_r4to8_m8.jsonl`) ran in 399.16s with
  8 workers, producing 230 candidate continuations across 15 labeled records; one selected state
  (`0410021`, state 38) produced no probe output and should be treated as a data-hygiene miss for
  future pool selection. Practical high-confidence best-vs-heuristic signals appeared on 6/15
  labeled states. The r4/min-rollout adaptive filter selected 3 solver-confirmation candidates:
  ante-3 `buy -> open_pack`, ante-2 `end_shop -> buy`, and ante-3 `sell -> buy`
  (`.data/phase8_basic_explore_diverse16c_deepen_candidates_minr4.jsonl`). Focused solver
  confirmation
  (`.data/phase8_solver_confirm_basic_explore_diverse16c_top3_focused_r4_m8.jsonl`) completed all
  3 in 262.94s with 3 workers. One candidate survived: ante-2 `end_shop -> buy`, mean advantage
  `+0.912`, SEM `0.688`, LCB `+0.224`, positive sample rate `0.75`
  (`.data/phase8_solver_confirmed_positive_labels_diverse16c_top3_minr4.jsonl`). The ante-3
  `buy -> open_pack` candidate was solver-rejected (mean `-2.478`, LCB `-3.412`), and the ante-3
  `sell -> buy` candidate stayed mean-positive but ambiguous (mean `+0.308`, LCB `-0.174`).
  Combined three-pool funnel: 47 fast-explored records -> 9 solver-confirmed candidates -> 3
  solver-confirmed positive labels.
- **Confidence-aware ranker target wired:** extended `ShopRankerExample`/batches with
  candidate-vs-baseline paired confidence fields (`advantage_lcbs`, `advantage_ucbs`,
  positive rates, rollout counts) and added `confidence_advantage_tie_mse`. This loss keeps
  confidence-supported positive/negative candidate-minus-baseline advantages but collapses
  uncertain intervals to a zero/tie target, which matches the "several viable early paths"
  problem better than mean-only advantage regression. Train and repeated-split scripts now accept
  the loss and report `confidence_advantage_label_summary` counts. Auditing the 9 solver-confirmed
  comparison records from the three diverse funnels gives a balanced tiny label set at margin
  `0.10`: 3 positive, 3 negative, and 3 ambiguous candidate-vs-baseline labels. A tiny repeated
  split smoke
  (`.data/phase8_ranker_sweep_solver_confirm_9_confidence_advantage_tie_mse.metrics.json`) runs
  end-to-end but is not deployable: attention still trails the heuristic on held-out regret
  (`0.829` vs `0.543`) and has negative mean lift (`-0.286` at threshold `0.0`, `-0.211` at
  threshold `0.10`). This confirms the target plumbing, not model strength. Focused tests pass
  (`112 passed`).
- **Fourth non-overlapping diverse-16 funnel and 15-label sweep:** selected the final non-overlap
  slice of the 64-state capture pool,
  `.data/phase8_capture_pool_v3_64_targeted_diverse_16d.jsonl`, excluding the prior three
  diverse pools. The remaining states are less varied by necessity (`buy=12`, `sell=4` heuristic
  actions) but remain ante-balanced (`8/8`). Basic exploration
  (`.data/phase8_sequential_baseline_probe_diverse16d_basic_r4to8_m8.jsonl`) completed 16/16
  records in 398.06s with 8 workers, produced 268 continuations, and had no skipped records. It
  selected 6 r4-supported solver-confirmation candidates:
  `buy -> open_pack` x2, `buy -> end_shop` x2, `sell -> end_shop`, and `sell -> open_pack`.
  Focused solver confirmation
  (`.data/phase8_solver_confirm_basic_explore_diverse16d_top6_focused_r4_m8.jsonl`) completed all
  6 in 291.76s with 6 workers. One candidate survived: ante-2 `buy -> end_shop`, mean advantage
  `+0.743`, SEM `0.546`, LCB `+0.198`, positive rate `0.75`
  (`.data/phase8_solver_confirmed_positive_labels_diverse16d_top6_minr4.jsonl`). One candidate was
  solver-rejected by negative UCB and four remained ambiguous. Combined four-pool funnel: 63
  fast-explored records -> 15 solver-confirmed candidates -> 4 confirmed positives. The
  confidence-aware label mix is now 4 positive, 4 negative, and 7 ambiguous candidate-vs-baseline
  labels at margin `0.10`. A 15-example confidence-aware split sweep
  (`.data/phase8_ranker_sweep_solver_confirm_15_confidence_advantage_tie_mse.metrics.json`)
  improves over the 9-example smoke but still does not clear deployment gates: mean encoder regret
  `0.578` vs heuristic `0.480`, attention regret `0.638` vs heuristic `0.480`; near-best/top-1
  are higher than the heuristic, but mean lift remains negative (`-0.098` mean, `-0.158`
  attention at threshold `0.0`) and harmful override rate is still too high (`0.333` mean,
  `0.262` attention). Verdict: the target is shaped correctly, but the model is still data-starved.
- **Fresh 128-state capture pool and selector balance fix:** generated
  `.data/phase8_capture_pool_v3_128_fresh.jsonl` from a new seed range (`seed_offset=420000`,
  `seed_count=256`) in 187.26s with 8 workers. It selected 128 capture-only shop states with exact
  64/64 source and ante balance from 1,024 captured / 1,001 deduped states. The heuristic-action
  distribution is much richer than the exhausted 64-state pool (`buy=52`, `sell=35`,
  `end_shop=23`, `reroll=7`, `open_pack=6`, `use_consumable=5`) with 70/128 solver heuristic
  actions outside the focused `buy/open_pack/end_shop` candidate set. While selecting the first
  fresh subset, found that `phase8_select_shop_state_pool.py` balanced only full tuple groups,
  which could still skew marginal fields such as ante. Replaced it with a greedy marginal-field
  balancer and verified the selector test. The first fresh targeted slice
  (`.data/phase8_capture_pool_v3_128_fresh_targeted_diverse_16a.jsonl`) is now exact 8/8 source,
  exact 8/8 ante, and spans all six heuristic action types (`buy=2`, `end_shop=2`, `open_pack=3`,
  `reroll=3`, `sell=3`, `use_consumable=3`). Fast basic exploration
  (`.data/phase8_sequential_baseline_probe_fresh128_diverse16a_basic_r4to8_m8.jsonl`) completed
  16/16 records in 393.99s with 8 workers and no skipped records. Only one r4-supported candidate
  passed the adaptive filter (`use_consumable -> end_shop`), and focused solver confirmation
  (`.data/phase8_solver_confirm_basic_explore_fresh128_diverse16a_top1_focused_r4_m8.jsonl`)
  made it ambiguous: mean advantage `+0.025`, SEM `0.847`, LCB `-0.823`. Combined confidence label
  pool is now 16 solver-confirmed comparisons: 4 positive, 4 negative, 8 ambiguous at margin
  `0.10`. The 16-example confidence-aware sweep
  (`.data/phase8_ranker_sweep_solver_confirm_16_confidence_advantage_tie_mse.metrics.json`) is
  still not deployable, but trends better on the mean encoder: regret `0.486` vs heuristic
  `0.408`, near-best@0.05 `0.624` vs heuristic `0.510`, top-1 `0.600` vs heuristic `0.371`, and
  mean lift `-0.079`. Attention is worse on regret/lift. Next scale move: select additional
  non-overlapping 16-state slices from the fresh 128 pool and keep accumulating solver-confirmed
  positives/negatives/ambiguities before trusting the neural override.
- **Fresh 128 slice B adds no-override supervision:** selected
  `.data/phase8_capture_pool_v3_128_fresh_targeted_diverse_16b.jsonl`, excluding fresh slice A.
  The improved selector again produced exact 8/8 source and ante balance and covered all six
  heuristic action types (`buy=3`, `end_shop=3`, `open_pack=2`, `reroll=2`, `sell=4`,
  `use_consumable=2`). Fast basic exploration
  (`.data/phase8_sequential_baseline_probe_fresh128_diverse16b_basic_r4to8_m8.jsonl`) completed
  16/16 records in 382.33s with 8 workers, producing 300 continuations and strong cheap evidence:
  practical high-confidence best-vs-heuristic on 11/16 states. The adaptive r4 filter selected 8
  solver-confirmation candidates across six heuristic action types. Focused solver confirmation
  (`.data/phase8_solver_confirm_basic_explore_fresh128_diverse16b_top8_focused_r4_m8.jsonl`)
  completed all 8 in 315.86s with 8 workers but produced zero confirmed positives: two candidates
  were rejected via negative UCB, five were ambiguous/max-rollouts, and one timed out after three
  paired samples. This is still useful data: combined confirmed comparisons are now 24 total with
  4 positive, 6 negative, and 14 ambiguous labels at margin `0.10`. The 24-example confidence
  sweep (`.data/phase8_ranker_sweep_solver_confirm_24_confidence_advantage_tie_mse.metrics.json`)
  moved further toward safe behavior but still does not promote: mean encoder regret `0.328` vs
  heuristic `0.286`, near-best@0.05 `0.630` vs `0.589`, top-1 `0.612` vs `0.571`, mean lift
  `-0.042`, override rate `0.260`, harmful override rate `0.286`. Attention is worse. The key
  takeaway is that cheap exploration is over-suggesting positives on some fresh states, and solver
  confirmation is correctly turning those into negative/tie labels for the conservative override
  model.
- **Fresh 128 slice C adds one real positive and crosses small validation lift:** selected
  `.data/phase8_capture_pool_v3_128_fresh_targeted_diverse_16c.jsonl`, excluding fresh slices A
  and B. The improved selector again held exact 8/8 source and ante balance, with heuristic-action
  coverage of `buy=4`, `end_shop=5`, `open_pack=1`, `reroll=2`, and `sell=4`; the remaining pool
  has no `use_consumable` baseline states and only one `open_pack` baseline state left. Fast basic
  exploration (`.data/phase8_sequential_baseline_probe_fresh128_diverse16c_basic_r4to8_m8.jsonl`)
  completed 16/16 records in 463.52s with 8 workers but was slower/noisier than slice B, with 45
  state timeouts and practical high-confidence best-vs-heuristic signals on 6/16 states. The
  adaptive r4 filter selected 3 solver-confirmation candidates. Focused solver confirmation
  (`.data/phase8_solver_confirm_basic_explore_fresh128_diverse16c_top3_focused_r4_m8.jsonl`)
  completed all 3 in 212.54s with 3 workers and produced one confirmed positive: ante-3
  `end_shop -> open_pack`, mean advantage `+1.493`, LCB `+1.124`, positive rate `1.0`
  (`.data/phase8_solver_confirmed_positive_labels_fresh128_diverse16c_top3_minr4.jsonl`). The
  combined solver-confirmed pool is now 27 candidate-vs-baseline records with 5 positive, 6
  negative, and 16 ambiguous confidence labels at margin `0.10`. The 27-example sweep
  (`.data/phase8_ranker_sweep_solver_confirm_27_confidence_advantage_tie_mse.metrics.json`) is the
  first aggregate positive-lift result: mean encoder regret `0.302` vs heuristic `0.343`, mean
  lift `+0.041`, near-best@0.05 `0.661` vs `0.556`, and top-1 `0.627` vs `0.474`. Attention is
  barely positive on lift (`+0.004`). This is progress but not promotion-ready: mean encoder
  harmful override rate is still `0.298` and attention harmful override rate is `0.255`.
- **Fresh 128 slice D adds ambiguity/no-override calibration:** selected
  `.data/phase8_capture_pool_v3_128_fresh_targeted_diverse_16d.jsonl`, excluding fresh slices A/B/C.
  Source and ante stayed exact 8/8, but the remaining heuristic-action mix narrowed to `buy=5`,
  `end_shop=5`, and `sell=6`. Fast basic exploration
  (`.data/phase8_sequential_baseline_probe_fresh128_diverse16d_basic_r4to8_m8.jsonl`) completed
  16/16 records in 394.31s with 8 workers and only 3/16 practical high-confidence
  best-vs-heuristic states. The adaptive filter selected 2 candidates with strong cheap LCBs, but
  focused solver confirmation
  (`.data/phase8_solver_confirm_basic_explore_fresh128_diverse16d_top2_focused_r4_m8.jsonl`) made
  both ambiguous/max-rollouts, producing zero confirmed positives
  (`.data/phase8_solver_confirmed_positive_labels_fresh128_diverse16d_top2_minr4.jsonl`, empty by
  design). The combined pool is now 29 records with 5 positive, 6 negative, and 18 ambiguous
  labels at margin `0.10`. The 29-example sweep
  (`.data/phase8_ranker_sweep_solver_confirm_29_confidence_advantage_tie_mse.metrics.json`) keeps
  mean encoder aggregate lift barely positive (`+0.018` at threshold `0.0`, `+0.027` at threshold
  `0.10`) but raises raw harmful override rate to `0.335`; the `0.10` threshold reduces harmful
  override rate to `0.271`. Verdict: slice D improves no-override calibration evidence, but the
  ranker is still not safe enough to deploy.
- **Fresh 128 slice E adds another strong pack positive and better safety metrics:** selected
  `.data/phase8_capture_pool_v3_128_fresh_targeted_diverse_16e.jsonl`, excluding fresh slices A-D.
  Source and ante stayed exact 8/8; the selector saw only `buy=38`, `end_shop=8`, and `sell=18`
  baselines remaining, and chose `buy=6`, `end_shop=5`, `sell=5`. Fast basic exploration
  (`.data/phase8_sequential_baseline_probe_fresh128_diverse16e_basic_r4to8_m8.jsonl`) completed
  16/16 records in 490.21s with 8 workers and found practical high-confidence best-vs-heuristic
  signals on 6/16 states, but only one candidate survived the stricter r4 filter. Focused solver
  confirmation (`.data/phase8_solver_confirm_basic_explore_fresh128_diverse16e_top1_focused_r4_m8.jsonl`)
  completed in 147.10s and confirmed a strong positive: ante-3 `buy -> open_pack`, mean advantage
  `+1.842`, LCB `+1.470`, positive rate `1.0`
  (`.data/phase8_solver_confirmed_positive_labels_fresh128_diverse16e_top1_minr4.jsonl`). The
  combined pool is now 30 records with 6 positive, 6 negative, and 18 ambiguous labels at margin
  `0.10`. The 30-example sweep
  (`.data/phase8_ranker_sweep_solver_confirm_30_confidence_advantage_tie_mse.metrics.json`) has
  positive aggregate lift for both encoders (`+0.042` mean, `+0.048` attention), but raw override
  behavior remains unsafe. Added absolute covered-state rates to the override metrics so we can
  distinguish conditional harm among overrides from harm per shop state. The mean-only threshold
  sweep (`.data/phase8_ranker_sweep_solver_confirm_30_mean_thresholds.metrics.json`) shows the
  tradeoff clearly: threshold `0.0` gives lift `+0.042` but harmful covered rate `0.179`;
  threshold `0.5` cuts harmful covered rate to `0.029` but turns lift negative (`-0.013`). Verdict:
  the model is learning real signal, but calibration is not deployment-safe yet.
- **Train-calibrated gate check:** added train-side threshold selection to
  `scripts/phase8_ranker_split_sweep.py`, with a configurable harmful-covered-rate cap
  (`--calibration-max-harmful-covered-rate`). This avoids picking an override threshold by peeking
  at validation. On the 30-record mean-encoder sweep with cap `0.05`
  (`.data/phase8_ranker_sweep_solver_confirm_30_mean_thresholds_calibrated.metrics.json`), the
  train-selected thresholds averaged `0.093` and did not transfer safely: validation lift was only
  `+0.007`, positive in 2/7 splits, and harmful covered rate stayed `0.163`. This is a useful
  negative result: the ranker has signal, but its confidence scores are not calibrated enough for
  a deployment gate chosen from current training labels.
- **Fresh2 128-state pool and first slice:** generated
  `.data/phase8_capture_pool_v3_128_fresh2.jsonl` from seed offset `430000` with 8 capture workers
  and 8 collect workers. It selected 128 exact-balanced source/ante states from 1,024 captured /
  986 deduped states in 270.25s. The first targeted slice
  (`.data/phase8_capture_pool_v3_128_fresh2_targeted_diverse_16a.jsonl`) restored broad heuristic
  action coverage: `buy=3`, `end_shop=3`, `open_pack=3`, `reroll=3`, `sell=3`, `use_consumable=1`,
  with exact 8/8 source and ante balance. Fast basic exploration
  (`.data/phase8_sequential_baseline_probe_fresh2_diverse16a_basic_r4to8_m8.jsonl`) completed
  16/16 records in 385.00s with 8 workers and found practical high-confidence best-vs-heuristic
  signals on 8/16 states. The r4 filter selected 4 solver-confirmation candidates. Focused solver
  confirmation (`.data/phase8_solver_confirm_basic_explore_fresh2_diverse16a_top4_focused_r4_m8.jsonl`)
  completed in 334.01s with 4 workers and produced one positive, one negative, one ambiguous, and
  one partial/timed-out candidate. The confirmed positive was an ante-3 pack-target choice
  (`open_pack -> open_pack`, likely a different pack index), mean advantage `+2.223`, LCB
  `+0.831`, positive rate `0.75`
  (`.data/phase8_solver_confirmed_positive_labels_fresh2_diverse16a_top4_minr4.jsonl`). The
  combined pool is now 34 records with 7 positive, 7 negative, and 20 ambiguous labels. The
  34-record sweep (`.data/phase8_ranker_sweep_solver_confirm_34_confidence_advantage_tie_mse.metrics.json`)
  is the strongest yet: mean encoder raw lift `+0.143` with harmful covered rate `0.109`; attention
  raw lift `+0.132` with harmful covered rate `0.071`. The train-calibrated mean gate under a
  `0.05` train harmful-covered cap now transfers much better than the 30-record run: validation
  lift `+0.126` and harmful covered rate `0.048`. Still do not deploy: calibrated mean is positive
  in only 3/7 splits, attention calibrated lift is only `+0.055`, and the label count remains tiny.
- **Fresh2 slice B adds no-override calibration and keeps neural lift positive:** selected a
  second non-overlapping fresh2 targeted slice
  (`.data/phase8_capture_pool_v3_128_fresh2_targeted_diverse_16b.jsonl`) with exact 8/8
  source/ante balance and 10/16 heuristic actions outside the focused candidate family. Fast basic
  exploration (`.data/phase8_sequential_baseline_probe_fresh2_diverse16b_basic_r4to8_m8.jsonl`)
  completed 16/16 records in 402.21s with 8 workers and found practical high-confidence
  best-vs-heuristic signals on 8/16 states. The r4 filter selected three ante-2 `end_shop`
  candidates against buy/open-pack baselines, but focused solver confirmation
  (`.data/phase8_solver_confirm_basic_explore_fresh2_diverse16b_top3_focused_r4_m8.jsonl`) made
  all three ambiguous: zero high-confidence positives, zero high-confidence negatives, and all
  stopped at max rollouts. Treat this as useful caution data: the cheap explorer still overvalues
  early shop-skipping proposals. The combined pool is now 37 records with 7 positive, 7 negative,
  and 23 ambiguous confidence labels at margin `0.10`. The 37-record sweep
  (`.data/phase8_ranker_sweep_solver_confirm_37_confidence_advantage_tie_mse.metrics.json`) keeps
  aggregate neural lift positive. Attention is strongest: regret `0.376` vs heuristic `0.523`,
  raw lift `+0.146` positive in 6/7 splits, and threshold `0.25` gives lift `+0.114` with harmful
  covered rate `0.079`. The train-calibrated attention gate is positive in 5/7 splits with lift
  `+0.118`, but held-out harmful covered rate is still `0.102`; calibration is improving, not
  deployment-safe.
- **Fresh2 slice C plus build-forward filtering:** added `--candidate-action-types` and
  `--exclude-candidate-action-types` to `scripts/phase8_select_deepening_states.py` so expensive
  solver confirmation can avoid known-noisy cheap proposal families when desired. The selector
  test now covers excluding an `end_shop` opportunity while retaining a build-forward candidate.
  Slice C (`.data/phase8_capture_pool_v3_128_fresh2_targeted_diverse_16c.jsonl`) stayed exact 8/8
  source/ante balanced, but the remaining fresh2 pool has narrowed to `buy/end_shop/open_pack/sell`
  baselines. Fast exploration
  (`.data/phase8_sequential_baseline_probe_fresh2_diverse16c_basic_r4to8_m8.jsonl`) labeled 15/16
  records in 414.65s with 8 workers and had only 2/15 practical high-confidence override states.
  The new build-only filter selected one ante-3 `end_shop -> buy` candidate. Solver confirmation
  (`.data/phase8_solver_confirm_basic_explore_fresh2_diverse16c_top1_buildonly_focused_r4_m8.jsonl`)
  made it mean-positive but ambiguous: mean advantage `+1.092`, SEM `1.295`, LCB `-0.204`. The
  combined pool is now 38 records with 7 positive, 7 negative, and 24 ambiguous labels. The
  38-record sweep (`.data/phase8_ranker_sweep_solver_confirm_38_confidence_advantage_tie_mse.metrics.json`)
  keeps positive neural lift. Attention has regret `0.310` vs heuristic `0.465`, raw lift `+0.156`
  positive in 5/7 splits, and train-calibrated lift `+0.124` positive in 5/7 splits with harmful
  covered rate `0.081`. This is an improvement over the 37-record calibrated harm, but still not
  below the desired `0.05` deployment cap.
- **Fresh2 slice D shows cheap pack-open overconfidence:** selected
  `.data/phase8_capture_pool_v3_128_fresh2_targeted_diverse_16d.jsonl`, excluding fresh2 slices
  A-C. Source/ante balance stayed exact 8/8, but the remaining pool narrowed further to
  `buy=45`, `end_shop=21`, and `sell=14` seen baselines. Fast basic exploration
  (`.data/phase8_sequential_baseline_probe_fresh2_diverse16d_basic_r4to8_m8.jsonl`) completed
  16/16 records in 385.55s with 8 workers and looked rich under the cheap proposer: practical
  high-confidence best-vs-heuristic on 7/16 states. The build-only filter selected three
  `open_pack` candidates against `end_shop`/`sell` baselines, with mean cheap LCB `+0.436`.
  Focused solver confirmation
  (`.data/phase8_solver_confirm_basic_explore_fresh2_diverse16d_top3_buildonly_focused_r4_m8.jsonl`)
  made all three ambiguous: mean advantage `+0.435`, SEM `1.374`, LCB `-0.939`, zero
  high-confidence positives. The combined pool is now 41 records with 7 positive, 7 negative, and
  27 ambiguous labels. The 41-record sweep
  (`.data/phase8_ranker_sweep_solver_confirm_41_confidence_advantage_tie_mse.metrics.json`) stayed
  positive but weakened: mean encoder regret `0.322` vs heuristic `0.385`, raw lift `+0.062`, and
  calibrated lift `+0.033`; attention regret `0.334` vs heuristic `0.385`, raw lift `+0.051`, and
  calibrated lift `+0.008`. Harmful covered rates remain above cap. Conclusion: the cheap proposer
  is overconfident on pack-open tempo too; these ambiguity labels improve caution but do not move
  the deployment gate forward.
- **Cheap-vs-solver proposal audit and SEM gate:** added
  `scripts/phase8_deepening_confirmation_audit.py` to join cheap deepening proposals against
  focused solver confirmations and report filter precision. Added optional `--max-sem` and
  `--min-lcb-sem-ratio` filters to `phase8_select_deepening_states.py`; focused tests pass
  (`7 passed`). The 41-record audit
  (`.data/phase8_deepening_confirmation_audit_41.metrics.json`) shows why LCB alone wasted solver
  time: solver-positive, solver-negative, and ambiguous proposals have overlapping cheap LCBs
  (`0.749`, `0.656`, `0.610` means), while cheap SEM separates them better (`0.487`, `0.748`,
  `0.842`). Retrospective filter precision: `max_sem=0.45` keeps 5 proposals with 4 positives,
  0 negatives, and 1 ambiguous (`0.80` positive precision); `max_sem=0.55` drops to 4/8 positives,
  and `max_sem=0.80` admits 5 negatives. Applying `max_sem=0.45` to fresh2 A-D build-forward
  cheap slices selects zero candidates, which would have saved the recent wasted solver
  confirmations. Use strict SEM gating for positive-label acquisition; relax it only when
  deliberately collecting no-override/ambiguous calibration data.
- **Retrospective SEM-gated confirmation and loader merge fix:** added `--exclude-records` to
  `phase8_select_deepening_states.py` so already-confirmed focused candidates are not selected
  again. Applying the strict `max_sem=0.45` build-forward filter across all paid cheap probes
  found only 3 still-unconfirmed candidates
  (`.data/phase8_allcheap_unconfirmed_buildonly_sem045_minr4.jsonl`): one `buy`, two
  `open_pack`, all ante 3, with mean cheap LCB `+0.434`. Focused solver confirmation
  (`.data/phase8_solver_confirm_allcheap_unconfirmed_buildonly_sem045_top3_focused_r4_m8.jsonl`)
  finished in 149.13s with 3 workers and produced 1 confirmed positive, 2 ambiguous, and 0
  negatives. The positive was seed `0420075`, state `39`, `sell -> open_pack`, mean advantage
  `+1.145`, LCB `+0.603`, positive rate `0.75`. Fixed the ranker JSONL loader so multiple
  focused confirmations for the same `(source_bot, seed, state_index)` merge new candidate
  actions instead of silently dropping later rows; focused tests pass (`30 passed`). The merged
  44-candidate-label sweep
  (`.data/phase8_ranker_sweep_solver_confirm_44_merged_confidence_advantage_tie_mse.metrics.json`)
  now has 8 positive, 7 negative, and 29 ambiguous candidate labels across 42 unique state
  examples. Attention is best on raw validation lift (`+0.226`, 6/7 positive runs) and calibrated
  lift (`+0.151`, 6/7), but calibrated harmful covered rate is still `0.149`, so the ranker remains
  a data/search prior rather than a deployable shop override.
- **Two more strict-SEM positives and the 46-label sweep:** re-ran the all-cheap strict-SEM
  selector after excluding the 44-label confirmations. It found two additional alternate pack
  choices on already-touched ante-3 states
  (`.data/phase8_allcheap_unconfirmed_buildonly_sem045_after44.jsonl`), both weaker by cheap LCB
  but still low-SEM. Focused solver confirmation
  (`.data/phase8_solver_confirm_allcheap_unconfirmed_buildonly_sem045_after44_focused_r4_m8.jsonl`)
  finished in 113.44s with 2 workers and confirmed both as positives: seed `0410006`, state `48`,
  `end_shop -> open_pack` at mean advantage `+0.871`; and seed `0420020`, state `50`,
  `sell -> open_pack` at mean advantage `+1.792`. Both stopped by `positive_lcb` at 4 paired
  rollouts. The merged 46-candidate-label sweep
  (`.data/phase8_ranker_sweep_solver_confirm_46_merged_confidence_advantage_tie_mse.metrics.json`)
  now has 10 positive, 7 negative, and 29 ambiguous candidate labels across the same 42 unique
  state examples. Mean encoder calibrated lift improved to `+0.199` and is positive in 7/7 split
  runs, with calibrated harmful covered rate down to `0.115`. Attention raw lift is highest
  (`+0.258`), but attention calibrated harm is worse (`0.170`). This is the strongest neural
  ranker evidence so far, but still not a deployable override because the covered harm is more
  than 2x the intended `0.05` cap.
- **Fresh2 slice E and action-family split lesson:** after the 46-label confirmations, the old
  paid cheap pool was exhausted under strict build-forward `max_sem=0.45` selection
  (`.data/phase8_allcheap_unconfirmed_buildonly_sem045_after46.metrics.json` selected zero).
  Selected a new balanced fresh2 slice E
  (`.data/phase8_capture_pool_v3_128_fresh2_targeted_diverse_16e.jsonl`): exact 8/8 source,
  exact 8/8 ante, heuristic mix `buy=5`, `end_shop=6`, `sell=5`. Fast basic exploration completed
  16 records in 385.59s with 8 workers and found 5/16 practical high-confidence states
  (`.data/phase8_sequential_baseline_probe_fresh2_diverse16e_basic_r4to8_m8.jsonl`), but strict
  build-forward SEM selection found zero candidates. Allowing `end_shop` found one strong-looking
  cheap candidate, seed `0430154` state `50`, `buy -> end_shop`, cheap mean `+1.550`, LCB `+1.154`;
  focused solver confirmation rejected it cleanly: buy mean `7.518`, end_shop mean `5.414`,
  candidate stopped by `negative_ucb`
  (`.data/phase8_solver_confirm_fresh2_diverse16e_safe_sem045_top1_focused_r4_m8.jsonl`). The
  resulting 47-label all-action sweep adds that as a negative label (10 positive, 8 negative,
  29 ambiguous across 43 states) but weakens calibrated lift; the build-forward-filtered sweep
  has 33 labels across 29 states and still leaves harmful covered rates around `0.106-0.149`.
  Conclusion: strict SEM is useful for buy/open-pack positive acquisition, but `end_shop`/skip
  economy proposals need a separate model/gate or a stronger confirmation filter.
- **Fresh2 slice F adds one more build-forward positive:** selected another non-overlapping
  fresh2 slice
  (`.data/phase8_capture_pool_v3_128_fresh2_targeted_diverse_16f.jsonl`) from the remaining 48
  states. Source and ante stayed exact 8/8; heuristic mix narrowed to `buy=6`, `end_shop=7`,
  `sell=3`. Fast cheap exploration
  (`.data/phase8_sequential_baseline_probe_fresh2_diverse16f_basic_r4to8_m8.jsonl`) completed 16
  records in 385.71s with 8 workers and 269 estimated candidate continuations. The strict
  build-forward SEM gate selected one ante-3 candidate, seed `0430019` state `38`,
  `end_shop -> open_pack`, cheap mean `+0.475`, SEM `0.271`, LCB `+0.203`, 6 cheap paired
  rollouts. Focused solver confirmation
  (`.data/phase8_solver_confirm_fresh2_diverse16f_buildonly_sem045_top1_focused_r4_m8.jsonl`)
  confirmed it as positive in 72.29s: mean advantage `+0.860`, SEM `0.402`, LCB `+0.458`,
  positive rate `1.0`. The merged all-action sweep is now 48 candidate labels across 44 state
  examples: 11 positive, 8 negative, 29 ambiguous
  (`.data/phase8_ranker_sweep_solver_confirm_48_merged_confidence_advantage_tie_mse.metrics.json`).
  It remains positive but unsafe: attention calibrated lift `+0.086` with harmful covered rate
  `0.127`. The build-forward-filtered sweep has 34 labels across 30 states, 10 positive / 5
  negative / 19 ambiguous
  (`.data/phase8_ranker_sweep_solver_confirm_48_merged_buildforward_confidence_advantage_tie_mse.metrics.json`);
  attention raw lift is `+0.249`, but calibrated lift is only `+0.064` with harmful covered rate
  `0.121`. Verdict unchanged: strict SEM acquisition is working, but deployment still needs more
  labels and better confidence calibration.
- **Fresh2 slice G adds caution labels and improves all-action calibration:** selected slice G
  from the final 32 fresh2 states
  (`.data/phase8_capture_pool_v3_128_fresh2_targeted_diverse_16g.jsonl`). Balance stayed exact
  8/8 source and ante, but the remaining pool was narrow (`buy=13`, `end_shop=3`). The 8-worker
  cheap pass (`.data/phase8_sequential_baseline_probe_fresh2_diverse16g_basic_r4to8_m8.jsonl`)
  completed 16 records in 428.08s with 225 estimated candidate continuations. The strict
  build-forward SEM gate selected two ante-3 `buy` candidates
  (`.data/phase8_basic_explore_fresh2_diverse16g_deepen_candidates_buildonly_sem045_minr4.jsonl`):
  seed `0430217`, state `38`, alternate `buy -> buy`, cheap mean `+0.562`, LCB `+0.253`; and seed
  `0430201`, state `41`, `end_shop -> buy`, cheap mean `+0.424`, LCB `+0.033`. Focused solver
  confirmation (`.data/phase8_solver_confirm_fresh2_diverse16g_buildonly_sem045_top2_focused_r4_m8.jsonl`)
  rejected the alternate buy as a negative label (candidate mean `7.386` vs heuristic mean
  `8.457`, advantage `-1.071`, `negative_ucb`) and left the `end_shop -> buy` candidate
  mean-positive but ambiguous (candidate mean `8.369` vs baseline `7.450`, advantage `+0.918`,
  max-rollouts). The merged all-action sweep now has 50 candidate labels across 46 state examples:
  11 positive, 9 negative, 30 ambiguous
  (`.data/phase8_ranker_sweep_solver_confirm_50_merged_confidence_advantage_tie_mse.metrics.json`).
  Despite no new confirmed positive, calibration improved: mean encoder calibrated lift `+0.150`
  with harmful covered `0.092`; attention calibrated lift `+0.158` with harmful covered `0.102`.
  The build-forward-filtered 50-label sweep is less stable
  (`.data/phase8_ranker_sweep_solver_confirm_50_merged_buildforward_confidence_advantage_tie_mse.metrics.json`),
  so the current best read is that caution labels help, but the dataset is still too small to hit
  the `0.05` harmful-covered safety cap.
- **Fresh2 slice H exhausted the pool; fresh3 pool started:** selected the final non-overlapping
  fresh2 slice
  (`.data/phase8_capture_pool_v3_128_fresh2_targeted_diverse_16h.jsonl`). Source and ante balance
  stayed exact 8/8, but all 16 baselines were `buy`, confirming the tail of fresh2 had lost action
  diversity. Cheap exploration
  (`.data/phase8_sequential_baseline_probe_fresh2_diverse16h_basic_r4to8_m8.jsonl`) completed 16
  records in 416.48s with 8 workers and looked tempting in aggregate, but strict build-forward
  `max_sem=0.45` selection found zero candidates
  (`.data/phase8_basic_explore_fresh2_diverse16h_deepen_candidates_buildonly_sem045_minr4.metrics.json`).
  No solver confirmation was run. Generated the next capture-only pool,
  `.data/phase8_capture_pool_v3_128_fresh3.jsonl`, from seed offset `440000`: 128 exact-balanced
  source/ante states, 1,024 captured / 994 deduped, 219.12s with 8 collect workers. The first
  targeted fresh3 slice
  (`.data/phase8_capture_pool_v3_128_fresh3_targeted_diverse_16a.jsonl`) restores action-family
  diversity: selected heuristic mix `buy=3`, `end_shop=3`, `open_pack=2`, `reroll=3`, `sell=3`,
  `use_consumable=2`, with exact 8/8 source and ante balance. Next: run the standard 8-worker
  cheap exploration on fresh3 slice A.
- **Fresh3 slice A adds ambiguity and improves attention safety:** the 8-worker cheap pass on
  fresh3 slice A (`.data/phase8_sequential_baseline_probe_fresh3_diverse16a_basic_r4to8_m8.jsonl`)
  completed 14 usable records in 435.50s and produced a stronger aggregate cheap signal than the
  exhausted fresh2 tail: mean best-vs-heuristic advantage `+1.367`, mean LCB `+0.671`, and
  practical high-confidence rate `0.571`. The strict build-forward `max_sem=0.45` gate selected
  two candidates
  (`.data/phase8_basic_explore_fresh3_diverse16a_deepen_candidates_buildonly_sem045_minr4.jsonl`):
  seed `0440085` state `22`, `open_pack -> buy`, cheap mean `+0.296`; and seed `0440204` state
  `38`, `reroll -> open_pack`, cheap mean `+0.726`. Solver confirmation
  (`.data/phase8_solver_confirm_fresh3_diverse16a_buildonly_sem045_top2_focused_r4_m8.jsonl`)
  made both confidence-ambiguous under the current label rule: `0440085` was mean-negative
  (`-0.569`) and `0440204` was mean-positive (`+0.840`), but neither cleared the interval gate.
  Adding them to the all-action sweep yields 52 candidate labels across 48 state examples: 11
  positive, 9 negative, 32 ambiguous
  (`.data/phase8_ranker_sweep_solver_confirm_52_merged_confidence_advantage_tie_mse.metrics.json`).
  Attention is now the safer deployment-style read: calibrated lift `+0.146`, positive in 6/7
  splits, calibrated harmful-covered `0.065`, and override rate `0.263`. That is closer to the
  `0.05` safety cap but still not below it. The build-forward-filtered sweep improved but remains
  less safe (`.data/phase8_ranker_sweep_solver_confirm_52_merged_buildforward_confidence_advantage_tie_mse.metrics.json`):
  attention calibrated lift `+0.177`, harmful-covered `0.119`. Net: fresh3 is producing useful
  caution labels, but the next acquisition pass still needs more clean positives or a stricter
  calibration rule before live shop overrides.
- **Fresh3 slice B shows why the strict gate matters:** selected slice B from the same fresh3
  pool while excluding slice A
  (`.data/phase8_capture_pool_v3_128_fresh3_targeted_diverse_16b.jsonl`): 16 states, exact 8/8
  source and ante balance, with heuristic mix `buy=3`, `end_shop=4`, `reroll=4`, `sell=4`,
  `use_consumable=1`. The 8-worker cheap pass
  (`.data/phase8_sequential_baseline_probe_fresh3_diverse16b_basic_r4to8_m8.jsonl`) completed
  all 16 records in 417.37s, with mean best-vs-heuristic advantage `+1.241`, LCB `+0.620`, and
  practical high-confidence rate `0.563`. But the strict build-forward `min_rollouts=4`,
  `max_sem=0.45` selector found zero candidates
  (`.data/phase8_basic_explore_fresh3_diverse16b_deepen_candidates_buildonly_sem045_minr4.metrics.json`):
  the strong-looking candidates were mostly timed out at 2-3 cheap paired rollouts. A separate
  min-3 near-miss pass found three open-pack candidates, then focused cheap deepening lifted all
  three to at least 4 rollouts
  (`.data/phase8_sequential_baseline_probe_fresh3_diverse16b_buildonly_sem045_minr3_focused_basic_r4to8_m8.jsonl`).
  Only one relaxed-after-focused candidate was worth solver confirmation, seed `0440079` state
  `43`, `end_shop -> open_pack`, cheap mean `+1.252`, LCB `+0.634`, SEM `0.618`
  (`.data/phase8_basic_explore_fresh3_diverse16b_deepen_candidates_buildonly_after_focused_lcb050_sem065_top1.jsonl`).
  Solver confirmation made it ambiguous rather than positive: mean advantage `+0.061`, SEM
  `0.790`, LCB `-0.729`
  (`.data/phase8_solver_confirm_fresh3_diverse16b_after_focused_lcb050_sem065_top1_focused_r4_m8.jsonl`).
  Adding this caution label gives the 53-label all-action sweep: 49 examples, 11 positive, 9
  negative, 33 ambiguous
  (`.data/phase8_ranker_sweep_solver_confirm_53_merged_confidence_advantage_tie_mse.metrics.json`).
  Mean encoder is now the closest deployment-style result so far: calibrated lift `+0.170`,
  positive in 6/7 splits, calibrated harmful-covered `0.062`, and override rate `0.282`. This is
  still above the `0.05` cap, but it is the best harm/lift tradeoff we have measured. The
  53-label build-forward-filtered sweep remains worse
  (`.data/phase8_ranker_sweep_solver_confirm_53_merged_buildforward_confidence_advantage_tie_mse.metrics.json`):
  calibrated harmful-covered is about `0.121-0.122`. Keep the strict min-4/SEM gate as the primary
  positive-acquisition rule; use relaxed-after-focused candidates only deliberately as caution data.
- **Fresh3 slice C adds a solver-confirmed build-forward positive:** selected slice C while
  excluding A/B (`.data/phase8_capture_pool_v3_128_fresh3_targeted_diverse_16c.jsonl`): 16
  states, exact 8/8 source and ante balance, with heuristic mix `buy=4`, `end_shop=4`,
  `reroll=4`, `sell=4`. The 8-worker cheap pass
  (`.data/phase8_sequential_baseline_probe_fresh3_diverse16c_basic_r4to8_m8.jsonl`) completed
  all 16 records in 404.98s. Aggregate cheap signal was weaker than B (mean best-vs-heuristic
  advantage `+0.884`, LCB `+0.051`, practical high-confidence rate `0.313`), and the strict
  `min_rollouts=4`, `max_sem=0.45` gate initially selected zero candidates. A single min-3
  near-miss (`sell -> open_pack`) strengthened after focused cheap deepening
  (`.data/phase8_sequential_baseline_probe_fresh3_diverse16c_buildonly_sem045_minr3_focused_basic_r4to8_m8.jsonl`):
  mean `+0.852`, SEM `0.280`, LCB `+0.572`, positive rate `1.0`. Solver confirmation
  (`.data/phase8_solver_confirm_fresh3_diverse16c_buildonly_after_focused_top1_focused_r4_m8.jsonl`)
  kept it positive: mean advantage `+0.564`, SEM `0.493`, LCB `+0.071`, positive rate `0.5`.
  Adding it gives the 54-label all-action sweep: 50 examples, 12 positive, 9 negative, 33
  ambiguous
  (`.data/phase8_ranker_sweep_solver_confirm_54_merged_confidence_advantage_tie_mse.metrics.json`).
  Utility improved: attention calibrated lift `+0.231`, mean calibrated lift `+0.197`, both
  positive in 7/7 splits. Harm is still above the cap: both encoders sit at calibrated
  harmful-covered `0.070`, so this is stronger but not safer than the 53-label mean gate (`0.062`).
  The 54-label build-forward-filtered sweep has high lift but high harm
  (`.data/phase8_ranker_sweep_solver_confirm_54_merged_buildforward_confidence_advantage_tie_mse.metrics.json`):
  calibrated harmful-covered `0.139-0.177`. Current read: more positives raise utility, while
  caution labels help safety; we need both, plus a better calibration rule, before deployment.
- **54-label calibration and conservative live A/B:** fixed threshold analysis on the 54-label
  all-action/safe-action sweep showed that train-side calibration is too aggressive, but a fixed
  baseline-margin threshold can clear the offline harm cap. At threshold `0.5`, attention keeps
  lift `+0.149` with harmful-covered `0.026`; threshold `1.0` keeps lift `+0.111` with
  harmful-covered `0.008`. Trained a full-data attention checkpoint
  (`.data/phase8_shop_ranker_solver_confirm_54_attention_confidence_advantage_tie_mse_full.pt`);
  in-sample all-label checks are clean at `0.5`, but this does not transfer online. On a held-out
  24-seed lane (`offset=540000`, `solver_shop_basic_play_bot` vs
  `solver_shop_basic_play_shop_ranker_bot`, safe actions, compare-baseline, ante 2-3, max 4
  candidates, one neural action per shop), margin `0.5` is neutral on wins but worse on mean ante
  (`2/24` vs `2/24`, mean ante `6.50 -> 6.38`, d_ante `-0.125`);
  margin `1.0` is worse (`2/24 -> 1/24`, d_ante `-0.542`). A regression trace showed several
  individually "safe" ante-2/3 overrides compounding across the run, so added
  `BALATRO_SHOP_RANKER_MAX_ACTIONS_PER_RUN` to cap total neural overrides, with a focused wrapper
  test (`python -m pytest -q tests\test_search_bot.py -k shop_ranker`, `11 passed`). The run cap
  helps but does not solve deployment: margin `0.5`, run cap `1` gains one win and loses none
  (`2/24 -> 3/24`) but still lowers mean ante (`6.50 -> 5.96`, d_ante `-0.542`);
  margin `1.0`, run cap `1` is also negative (`2/24 -> 1/24`, d_ante `-0.417`). Verdict:
  the ranker remains useful offline signal and a label-acquisition prior, but the current
  checkpoint is not a live shop override. Next work should target label/action-distribution
  mismatch, especially voucher/buy target kinds and repeated pack/economy overrides, before
  another online promotion attempt.
- **Action-kind filtering confirms target mismatch is real but not sufficient:** audited the
  54-label non-heuristic candidate pool by action kind: only 2 `buy/voucher` labels exist,
  compared with 14 `buy/card` and 24 `open_pack/pack`, so broad online `BUY` was under-supported.
  Added live `BALATRO_SHOP_RANKER_ACTION_KINDS` filtering and matching offline
  `--candidate-action-kinds` support in the ranker loader/train/sweep CLIs; focused tests pass
  (`python -m pytest -q tests\test_shop_ranker.py tests\test_search_bot.py -k "shop_ranker or candidate_action_kind or parse_action_kinds"`,
  `35 passed`). The exact card/pack safe-action offline sweep
  (`.data/phase8_ranker_sweep_solver_confirm_54_merged_safeactions_cardpack_confidence_advantage_tie_mse.metrics.json`)
  is more conservative: attention calibrated lift `+0.172`, harmful-covered `0.057`; at fixed
  threshold `0.5`, attention lift `+0.091` with harmful-covered `0.038`. Live A/B with
  `ACTION_KINDS=card,pack`, margin `0.5`, ante 2-3, one neural action per shop, and no run cap
  was positive on the first 24 seeds (`2/24 -> 3/24`, mean ante `6.50 -> 6.58`) but failed on the
  next 24 (`6/24 -> 2/24`, mean ante `6.54 -> 5.88`). Combined 48-seed read is negative:
  wins `8 -> 5`, mean ante about `6.52 -> 6.23`, better/same/worse `16/10/22`. Conclusion:
  excluding vouchers fixes one obvious mismatch but the model still does not generalize online.
  Next labels should be collected from live override-disagreement states, not only fresh random
  shop states: capture ranker-proposed `card/pack/end_shop` overrides that baseline would reject,
  solver-confirm them, and train specifically against those deployment-distribution mistakes.
- **Deployment-disagreement capture path is now wired:** added
  `scripts/phase8_ranker_override_capture.py` to follow `solver_shop_basic_play_bot` trajectories,
  ask the trained ranker for compare-baseline override proposals under the same live gates, and
  write `deepening_candidate_action_key` records that can flow directly into
  `phase8_sequential_baseline_probe.py --focus-deepening-candidate`. Also hardened the sequential
  probe so a captured focus action is replayable even if the regenerated candidate budget would
  omit it. Focused tests pass:
  `python -m pytest -q tests\test_phase8_ranker_override_capture.py tests\test_phase8_sequential_baseline_probe.py tests\test_search_bot.py -k "ranker_override_capture or sequential_baseline_probe or shop_ranker"`
  (`21 passed, 45 deselected`). A held-out 8-seed smoke at offset `560000` with the full 54-label
  attention checkpoint, card/pack kinds, margin `0.5`, ante 2-3, max 4 candidates, one neural
  action per shop/run captured 4 override disagreements in 53.89s
  (`.data/phase8_ranker_override_capture_smoke.jsonl`): 2 `buy/card`, 2 `open_pack/pack`. Full
  horizon solver confirmation with a 30s state cap was too expensive and skipped all 4; a shorter
  2-ante smoke confirmed the pipeline in 53.75s
  (`.data/phase8_ranker_override_capture_smoke_confirmed_h2.jsonl`) and joined all 4 proposals:
  0 positive, 1 negative, 3 ambiguous
  (`.data/phase8_ranker_override_capture_smoke_confirmed_h2_audit.metrics.json`). This is tiny and
  short-horizon, so do not train from it as-is, but it validates the next label-acquisition lane
  and shows the current ranker is confidently proposing deployment actions that solver
  confirmation does not yet endorse.
  Scaling the capture-only pass to 32 held-out seeds with 8 workers captured a 16-record queue in
  75.39s (`.data/phase8_ranker_override_capture_560000_32s16.jsonl`): 15 ante-2 states and 1
  ante-3 state, 6 `buy/card` proposals, 10 `open_pack/pack` proposals, and mean baseline margin
  `0.913`. Short 2-ante confirmation for all 16 finished in 72.48s with 8 workers
  (`.data/phase8_ranker_override_capture_560000_32s16_confirmed_h2.jsonl`) and joined as 2
  positive, 1 negative, 13 ambiguous
  (`.data/phase8_ranker_override_capture_560000_32s16_confirmed_h2_audit.metrics.json`). Treat
  this as triage only; the next expensive gate is deeper confirmation of this queue, especially
  the two short-horizon positives and any high-margin ambiguous pack overrides.
- **Deployment-disagreement deep confirmation adds caution labels, not positives:** selected the
  two h2-positive deployment-disagreement records for deeper confirmation
  (`.data/phase8_ranker_override_capture_560000_32s16_deepen_h2_pos.jsonl`): seed `0560008`
  `end_shop -> buy/card` and seed `0560006` `end_shop -> open_pack/pack`. The focused r4-to-r8,
  max-ante-8 confirmation took 224.52s with 2 workers and completed all 8 paired rollouts for both
  records
  (`.data/phase8_ranker_override_capture_560000_32s16_deepen_h2_pos_confirmed_r4_m8.jsonl`).
  Both became ambiguous/no-override caution labels: 0 positive, 0 negative, 2 ambiguous, with mean
  solver LCB `-1.599`
  (`.data/phase8_ranker_override_capture_560000_32s16_deepen_h2_pos_confirmed_r4_m8_audit.metrics.json`).
  Adding those two labels to the merged sweep gives 52 examples / 56 labels: 12 positive, 9
  negative, 35 ambiguous
  (`.data/phase8_ranker_sweep_solver_confirm_56_merged_deployment_confidence_advantage_tie_mse.metrics.json`).
  All-action calibration did not improve: attention calibrated lift `+0.111` with harmful-covered
  `0.099`; mean calibrated lift `+0.096` with harm `0.108`. The matching deployment-safe card/pack
  sweep (`.data/phase8_ranker_sweep_solver_confirm_56_merged_deployment_safeactions_cardpack_confidence_advantage_tie_mse.metrics.json`)
  keeps lift but remains unsafe under train-calibrated thresholds: attention calibrated lift
  `+0.129`, harmful-covered `0.097`; mean calibrated lift `+0.150`, harmful-covered `0.115`.
  Fixed attention threshold `0.5` is close but still above the cap (lift `+0.065`, harm `0.062`);
  threshold `1.0` clears harm (`0.018`) but is low-lift/unstable (`+0.029`, positive in 3/7
  splits). Conclusion: deployment-disagreement labels are the right data lane, but two caution
  labels are nowhere near enough. Scale this lane and prioritize deeper confirmations for
  high-margin ranker overrides before training another full-data live checkpoint.
- **58-label deployment sweep finds a plausible offline gate, but live A/B does not promote:**
  selected the next two short-horizon mean-positive deployment disagreements
  (`.data/phase8_ranker_override_capture_560000_32s16_deepen_h2_meanpos_next4.jsonl`) and ran the
  same focused r4-to-r8, max-ante-8 confirmation. Both stopped early by positive LCB after 6
  paired rollouts and became solver-confirmed positives: 2 positive, 0 negative, 0 ambiguous,
  mean solver LCB `+0.177`
  (`.data/phase8_ranker_override_capture_560000_32s16_deepen_h2_meanpos_next4_confirmed_r4_m8_audit.metrics.json`).
  Merging those with the two prior caution labels gives 54 all-action examples / 58 labels:
  14 positive, 9 negative, 35 ambiguous
  (`.data/phase8_ranker_sweep_solver_confirm_58_merged_deployment_confidence_advantage_tie_mse.metrics.json`).
  All-action attention recovers lift (`+0.207`) but validation harmful-covered is still high
  (`0.079`). The exact deployment-safe card/pack sweep has 52 examples / 56 labels
  (`.data/phase8_ranker_sweep_solver_confirm_58_merged_deployment_safeactions_cardpack_confidence_advantage_tie_mse.metrics.json`):
  attention calibrated lift `+0.177`, harm `0.075`; fixed attention threshold `0.5` is the first
  plausible offline gate, with lift `+0.133`, harmful-covered `0.025`, and 6/7 positive splits.
  Trained a full-data safe attention checkpoint
  (`.data/phase8_shop_ranker_solver_confirm_58_attention_safe_cardpack_confidence_advantage_tie_mse_full.pt`)
  and tested it live with baseline comparison, card/pack kinds only, ante 2-3, max 4 candidates,
  one neural action per shop/run, and baseline margin `0.5`. The first fresh 24-seed block
  improved (`4/24 -> 6/24`, mean ante `+0.042` at offset `580000`), but the second regressed
  (`3/24 -> 1/24`, mean ante `-0.208` at offset `590000`). Combined 48-seed read is not
  promotable: wins tie `7 -> 7`, mean ante `-0.083`, better/worse/same `13/18/17`, and win flips
  tie `5/5`. Verdict: labels are improving enough to generate useful proposals, but online
  generalization is still unstable. Do not promote this checkpoint; use it as a proposal/label
  acquisition model.
- **New label lane to add next: winning-trajectory backward reanalysis:** generate or capture runs
  that reach/win ante 8, keep full shop-state snapshots along the trajectory, then branch from the
  last shop choice across all legal alternatives and roll forward from that snapshot. This gives
  cheap, low-horizon late-game labels first, then can walk backward shop-by-shop into ante 7, 6,
  and earlier. It is not a replacement for early-game labels because winning trajectories have
  survivorship bias, so include near-wins/losses too, but it directly targets the late-game build
  planning problem without paying ante-1-to-8 rollout cost for every candidate.
- **Backward late-shop capture is now implemented and smoke-tested:** added
  `scripts/phase8_backward_shop_state_capture.py`, a capture-only generator for the backward
  reanalysis lane. It runs a bot on fresh seeds, stores real shop snapshots in memory, and writes
  the last N late shops only for trajectories that win or reach a requested terminal ante. Caps now
  prioritize winning and later-terminal records, covered by
  `tests/test_phase8_backward_shop_state_capture.py` (`2 passed`). A relaxed smoke proved the
  script writes records, then a real solver capture on 16 fresh seeds at offset `620000` produced
  14 ante-8 shop snapshots from 7 qualifying trajectories in 225.25s with 8 workers
  (`.data/phase8_backward_shops_solver_620000_16_late.jsonl`,
  `.data/phase8_backward_shops_solver_620000_16_late.metrics.json`). A tiny end-to-end label
  smoke on 2 of those snapshots completed in 54.70s with 2 workers and 26 candidate continuations
  (`.data/phase8_backward_shops_solver_620000_16_late_label_smoke.jsonl`). The smoke labels are
  not training quality (`r=2`, 1-ante horizon), but they prove the captured late shops flow through
  the existing `phase8_shop_candidate_dataset.py --input-records` multiworker labeler.
- **First backward late-shop labels and sweep:** labeled all 14 ante-8 snapshots from the
  `620000` pool with `r=4`, one-ante horizon, max 8 actions, and 8 workers
  (`.data/phase8_backward_shops_solver_620000_16_late_r4_h1_m8.jsonl`). The run completed 264
  candidate continuations in 523.31s. This near-win-heavy pool has real override signal: mean
  best-vs-heuristic advantage `+0.133`, mean LCB `+0.028`, high-confidence best-beats-heuristic
  rate `0.429`, and practical high-confidence override-candidate rate `0.143`. A second fresh
  capture on 32 seeds at offset `630000` produced 8 more ante-8 snapshots, all from winning
  trajectories
  (`.data/phase8_backward_shops_solver_630000_32_late.jsonl`), and labeling them took 181.40s for
  164 continuations
  (`.data/phase8_backward_shops_solver_630000_32_late_r4_h1_m8.jsonl`). Winning-run labels are
  more split-half stable (`0.75`) but flatter: mean best-vs-heuristic LCB `+0.006`, practical
  high-confidence overrides `0.0`. The combined 22-record backward-only sweep
  (`.data/phase8_ranker_sweep_backward_late_22_r4_h1_m8_confidence_advantage_tie_mse.metrics.json`)
  is learnable as regret supervision but not yet as a confident live policy: 85 candidate labels
  are 3 positive, 6 negative, 76 ambiguous. Attention beats the heuristic on held-out regret
  (`0.126` vs `0.164`) and near-best@0.05 (`0.643` vs `0.381`), but confidence calibration is
  still sparse. Mean encoder at fixed threshold `0.1` shows a tiny safe gate (`+0.043` lift,
  `0.0` harmful-covered, 6/7 positive splits); attention at threshold `0.1` is too suppressed
  (`-0.002` lift). Verdict: backward reanalysis is a viable late-game label lane, especially from
  near-wins, but needs a larger pool before training a checkpoint.
- **Near-win targeting added and validated:** added `--exclude-wins` to
  `scripts/phase8_backward_shop_state_capture.py` so the backward lane can deliberately collect
  ante-8 losses/near-misses instead of already-winning trajectories. Focused tests now cover both
  win-only and near-win qualification (`python -m pytest -q tests\test_phase8_backward_shop_state_capture.py`,
  `4 passed`). A fresh 32-seed near-win capture at offset `640000` found 3 qualifying ante-8
  losses and wrote 6 late-shop snapshots, while excluding 5 wins from the same block
  (`.data/phase8_backward_shops_solver_640000_32_late_nearwin.jsonl`). Labeling those 6 snapshots
  with the same `r=4`, h1, max-8 setup took 374.58s for 112 continuations
  (`.data/phase8_backward_shops_solver_640000_32_late_nearwin_r4_h1_m8.jsonl`) and produced the
  strongest backward signal yet: mean best-vs-heuristic advantage `+0.395`, mean LCB `+0.165`,
  oracle practical-positive rate `0.667`, and practical high-confidence override-candidate rate
  `0.333`. Adding these to the backward sweep gives 28 examples and 107 candidate labels: 7
  positive, 12 negative, 88 ambiguous
  (`.data/phase8_ranker_sweep_backward_late_28_r4_h1_m8_confidence_advantage_tie_mse.metrics.json`).
  Mean encoder now clearly beats heuristic regret (`0.162` vs `0.207`, wins 6/7) and near-best@0.05
  (`0.607` vs `0.357`), but confidence-gated overrides are not safe yet (`threshold=0.1` lift
  `+0.023`, harmful-covered `0.125`; threshold `0.25` nearly suppresses all lift). Attention
  regresses on the 28-record set (`0.213` vs `0.207`), so use the mean encoder for this late-game
  lane until the pool is much larger. Next target: collect more `--exclude-wins` ante-8 pools and
  then relabel selected high-signal states with deeper/r8 confirmation.
- **Backward deepening funnel validated:** selected high-signal states from the 28-record backward
  pool with `phase8_select_deepening_states.py` using candidate-minus-heuristic filters
  (`mean >= 0.10`, LCB `>= 0.05`, positive rate `>= 0.75`, max SEM `0.80`). The selector found 5
  actionable ante-8 states
  (`.data/phase8_backward_late_28_deepen_select_m010_lcb005_pr075_sem080.jsonl`) with mean cheap
  advantage `+0.628` and mean LCB `+0.325`. Deeper `r=8`, h1 confirmation completed all 5 in
  745.93s
  (`.data/phase8_backward_late_28_deepen_select_m010_lcb005_pr075_sem080_r8_h1_m8.jsonl`): every
  selected state had a non-heuristic best action, best-vs-heuristic practical positives were 5/5
  by mean, and 3/5 were practical high-confidence improvements. The exact cheap-selected
  proposals audited as 3 positive, 0 negative, 2 ambiguous
  (`.data/phase8_backward_late_28_deepen_select_m010_lcb005_pr075_sem080_r8_h1_m8.confirmation.json`).
  Verdict: cheap `r=4` labels are not reliable enough to deploy, but they are good enough to
  prioritize deeper late-game confirmations without poisoning the pool.
- **Second near-win block is mostly calibration/no-override data:** a fresh 32-seed capture at
  offset `650000` found 4 non-winning ante-8 trajectories and wrote 8 late-shop snapshots, while
  excluding 7 wins
  (`.data/phase8_backward_shops_solver_650000_32_late_nearwin.jsonl`). Cheap `r=4`, h1 labeling was
  much faster than the prior near-win block (144 continuations in 117.64s) but had little override
  signal: heuristic best rate `0.375`, heuristic within `0.10` on 7/8 states, mean
  best-vs-heuristic advantage `+0.038`, mean LCB `-0.039`, and zero high-confidence override
  candidates
  (`.data/phase8_backward_shops_solver_650000_32_late_nearwin_r4_h1_m8.jsonl`). Adding this
  calibration block gives a 36-record backward sweep
  (`.data/phase8_ranker_sweep_backward_late_36_r4_h1_m8_confidence_advantage_tie_mse.metrics.json`):
  135 candidate labels are 7 positive, 17 negative, 111 ambiguous. Mean encoder barely beats
  heuristic regret (`0.086` vs `0.089`), attention near-best improves (`0.631` vs `0.607`) but
  regret regresses (`0.091` vs `0.089`), and confidence-gated lift remains too small for live use.
  After excluding the 5 already deepened states, the strict selector found 0 remaining unconfirmed
  opportunities, so the next useful action is more fresh near-win/fringe capture, not more
  deepening of this same pool.
- **Third near-win block plus confirmed-label overlay:** a fresh 32-seed capture at offset
  `660000` found 6 non-winning ante-8 trajectories and wrote 12 late-shop snapshots, excluding
  3 wins (`.data/phase8_backward_shops_solver_660000_32_late_nearwin.jsonl`). Cheap `r=4`, h1
  labels took 230.04s for 232 continuations
  (`.data/phase8_backward_shops_solver_660000_32_late_nearwin_r4_h1_m8.jsonl`). This block has
  candidate signal but noisy winners: heuristic best rate `0.0`, heuristic within `0.10` only
  `0.25`, mean best-vs-heuristic advantage `+0.241`, but mean LCB `-0.071` and only `1/12`
  practical high-confidence best-vs-heuristic states. The expanded 48-record cheap sweep
  (`.data/phase8_ranker_sweep_backward_late_48_r4_h1_m8_confidence_advantage_tie_mse.metrics.json`)
  is not promotable: 181 candidate labels are 9 positive, 20 negative, 152 ambiguous; mean encoder
  regret regresses vs heuristic (`0.201` vs `0.193`), attention is nearly tied on regret
  (`0.193` vs `0.193`) while improving near-best (`0.420` vs `0.330`), and confidence gates remain
  harmful. The strict selector, after excluding the first 5 r8 confirmations, found one new
  `open_pack` over `end_shop` opportunity with cheap advantage `+0.826` and LCB `+0.410`
  (`.data/phase8_backward_late_48_deepen_select_unconfirmed_m010_lcb005_pr075_sem080.jsonl`).
  Focused `r=8` confirmation completed in 45.87s and confirmed it cleanly: best-vs-heuristic
  advantage `+0.786`, LCB `+0.501`, and exact proposal audit `1` positive / `0` negative / `0`
  ambiguous
  (`.data/phase8_backward_late_48_deepen_select_unconfirmed_m010_lcb005_pr075_sem080_r8_h1_m8.jsonl`).
  Fixed `examples_from_jsonl_paths` candidate merging so deeper duplicate candidate labels replace
  shallow duplicates instead of being skipped; the regression test is in
  `tests/test_shop_ranker.py`. With the 6 r8 confirmations overlaid, the 48-state sweep
  (`.data/phase8_ranker_sweep_backward_late_48_r4_plus_r8confirm_h1_m8_confidence_advantage_tie_mse.metrics.json`)
  improves the read but still is not a checkpoint source: mean encoder regret `0.153` vs heuristic
  `0.157` and attention `0.154` vs `0.157`, near-best improves to `0.438`/`0.446` vs heuristic
  `0.330`, but fixed confidence gates still do not show useful safe lift. After excluding all 6
  r8 confirmations, the strict selector finds zero remaining opportunities in the 48-state pool.
  Next: collect more fresh near-win/fringe blocks; do not spend more deepening on this exhausted
  pool unless the selection rule changes.
- **Keep winning late shops in the backward lane:** after review, changed the acquisition stance:
  do not filter out all wins. If a final shop turns a fragile ante-8 run into a win, those
  snapshots are exactly the positive late-build examples the model needs. The `670000`
  near-win-only block added 6 more ante-8 loss snapshots
  (`.data/phase8_backward_shops_solver_670000_32_late_nearwin_r4_h1_m8.jsonl`); cheap selection
  found 3 `open_pack` opportunities, but r8 confirmation made all three exact proposals
  ambiguous (`0` positive, `0` negative, `3` ambiguous), so they are calibration, not positives.
  The next block at offset `680000` intentionally omitted `--exclude-wins`, capturing 20 ante-8
  snapshots from 10 qualifying trajectories: 12 records from 6 wins and 8 records from 4
  near-wins
  (`.data/phase8_backward_shops_solver_680000_32_late_mixed.jsonl`). Fixed
  `phase8_shop_candidate_dataset.py` so backward metadata (`terminal_won`,
  `selection_reason`, terminal score/money, shops-from-terminal) survives relabeling and appears
  in metrics; the metadata-preserving relabel is
  `.data/phase8_backward_shops_solver_680000_32_late_mixed_r4_h1_m8_meta.jsonl`.
- **Mixed win/near-win selector found a clean winning-run positive:** extended
  `phase8_select_deepening_states.py` to carry `terminal_won` / `selection_reason` and balance by
  terminal outcome. Running the strict selector over the 74-state backward pool with
  `--balance-fields terminal_won,heuristic_action_type` found two strong opportunities, one from
  a win and one from a near-win
  (`.data/phase8_backward_late_74_deepen_select_balanced_terminal_m010_lcb005_pr075_sem080.jsonl`).
  Focused r8 confirmation kept both best actions clearly above the heuristic on mean
  (mean best-vs-heuristic advantage `+0.678`, LCB `+0.557`), but exact proposal audit split
  `1` positive / `0` negative / `1` ambiguous. The clean positive was from winning seed
  `0680010`: `open_pack` over heuristic `sell`, r8 mean advantage `+1.194`, LCB `+1.070`.
  This directly validates keeping wins in the lane. With all r8 confirmations overlaid, the
  74-state backward sweep
  (`.data/phase8_ranker_sweep_backward_late_74_mixed_r4_plus_r8confirm_h1_m8_confidence_advantage_tie_mse.metrics.json`)
  has 14 positive, 31 negative, and 237 ambiguous labels. Mean encoder now beats heuristic regret
  on every split (`0.070` vs `0.118`, 7/7 wins) and near-best@0.05 (`0.667` vs `0.536`), while
  attention is weaker (`0.095` regret, 5/7 wins). Fixed-threshold gate lift is still too small and
  harmful-covered is too high (`mean` threshold `0.1`: `+0.015` lift, `0.060` harm), so this is a
  better label pool but not yet a live checkpoint.
- **Second mixed block confirms winning-run positives but not deployable calibration:** after the
  74-state pool was exhausted under the strict selector, captured another mixed block at offset
  `690000` with wins included. The block produced 24 ante-8 snapshots from 12 qualifying
  trajectories: 14 records from 7 wins and 10 records from 5 near-wins
  (`.data/phase8_backward_shops_solver_690000_32_late_mixed.jsonl`). Cheap `r=4`, h1 labeling
  preserved terminal metadata and took 507.70s for 452 continuations
  (`.data/phase8_backward_shops_solver_690000_32_late_mixed_r4_h1_m8_meta.jsonl`). It was mostly
  tied/calibration data (heuristic within `0.10` on 18/24 states), but the balanced selector found
  two strong unconfirmed opportunities, both from winning trajectories and both `open_pack`
  candidates over heuristic `buy`/`end_shop`
  (`.data/phase8_backward_late_98_deepen_select_balanced_terminal_m010_lcb005_pr075_sem080.jsonl`).
  Focused r8 confirmation made both exact proposals positive, 2 positive / 0 negative / 0
  ambiguous, with mean best-vs-heuristic advantage `+0.507`, LCB `+0.255`
  (`.data/phase8_backward_late_98_deepen_select_balanced_terminal_m010_lcb005_pr075_sem080_r8_h1_m8.jsonl`).
  This is another direct win for keeping successful trajectories in the backward lane. The
  98-state overlay sweep
  (`.data/phase8_ranker_sweep_backward_late_98_mixed_r4_plus_r8confirm_h1_m8_confidence_advantage_tie_mse.metrics.json`)
  now has 20 positive, 44 negative, and 307 ambiguous labels. It still beats the heuristic on
  average regret (`mean` encoder `0.107` vs heuristic `0.131`; attention `0.112`), but the read is
  less stable than the 74-state pool (only 4/7 regret wins) and fixed gates are not safe
  (`mean` threshold `0.1`: lift `-0.004`, harm `0.071`). After excluding all r8 confirmations,
  the strict selector finds zero remaining opportunities in the 98-state pool. Next data step:
  keep mixed capture, but do not promote a checkpoint until gate calibration improves.
- **Third mixed block strengthens the "keep wins too" lane:** offset `700000` captured 24 ante-8
  snapshots from 12 qualifying trajectories, including 6 records from 3 wins and 18 records from
  9 near-wins
  (`.data/phase8_backward_shops_solver_700000_32_late_mixed.jsonl`). Cheap `r=4`, h1 labeling
  was signal-rich: heuristic within `0.10` on only 12/24 states, mean best-vs-heuristic advantage
  `+0.249`, mean LCB `+0.092`, and practical high-confidence best-vs-heuristic rate `0.25`
  (`.data/phase8_backward_shops_solver_700000_32_late_mixed_r4_h1_m8_meta.jsonl`). Adding this
  block to the prior pool and excluding all existing r8 confirmations left 5 strict unconfirmed
  opportunities, balanced as 2 from wins and 3 from near-wins
  (`.data/phase8_backward_late_122_deepen_select_balanced_terminal_m010_lcb005_pr075_sem080.jsonl`).
  Focused r8 confirmation kept all 5 best actions above the heuristic by mean, with 4/5 practical
  high-confidence, and exact proposal audit was 4 positive / 0 negative / 1 ambiguous
  (`.data/phase8_backward_late_122_deepen_select_balanced_terminal_m010_lcb005_pr075_sem080_r8_h1_m8.jsonl`).
  The 122-state overlay sweep now has 37 positive, 51 negative, and 376 ambiguous candidate
  labels
  (`.data/phase8_ranker_sweep_backward_late_122_mixed_r4_plus_r8confirm_h1_m8_confidence_advantage_tie_mse.metrics.json`).
  Both encoders beat heuristic regret on all seven seed splits: mean encoder `0.123` vs heuristic
  `0.156`, attention `0.119` vs heuristic `0.156`. Mean encoder is the safer current candidate:
  near-best@0.05 `0.536` vs heuristic `0.404`, top-1 `0.282` vs `0.146`, calibrated lift `+0.011`
  in 6/7 runs, and fixed threshold `0.1` lift `+0.017` in 7/7 runs. However, harmful covered rate
  at threshold `0.1` is still `0.086`, so this is progress in label quality and split stability,
  not a promotable live checkpoint yet.
- **Fourth mixed block was mostly calibration, not new strength:** offset `710000` captured 14
  ante-8 snapshots from 7 qualifying trajectories, including 4 records from 2 wins and 10 records
  from 5 near-wins
  (`.data/phase8_backward_shops_solver_710000_32_late_mixed.jsonl`). Cheap `r=4`, h1 labeling
  showed a flat/tied block: heuristic within `0.10` on 11/14 states, mean best-vs-heuristic
  advantage only `+0.058`, mean LCB `-0.058`, and zero practical high-confidence
  best-vs-heuristic cases
  (`.data/phase8_backward_shops_solver_710000_32_late_mixed_r4_h1_m8_meta.jsonl`). The expanded
  136-state selector found one marginal near-win `open_pack` over `end_shop` opportunity
  (`.data/phase8_backward_late_136_deepen_select_balanced_terminal_m010_lcb005_pr075_sem080.jsonl`),
  but r8 confirmation made the exact proposal ambiguous, 0 positive / 0 negative / 1 ambiguous
  (`.data/phase8_backward_late_136_deepen_select_balanced_terminal_m010_lcb005_pr075_sem080_r8_h1_m8.jsonl`).
  The 136-state overlay sweep has 37 positive, 58 negative, and 426 ambiguous labels
  (`.data/phase8_ranker_sweep_backward_late_136_mixed_r4_plus_r8confirm_h1_m8_confidence_advantage_tie_mse.metrics.json`).
  Both encoders still beat heuristic regret on average, but only 5/7 splits now: mean `0.129` vs
  heuristic `0.141`, attention `0.120` vs heuristic `0.141`. Calibration weakened (`mean`
  calibrated lift `+0.002`, threshold `0.1` harmful-covered `0.134`; attention calibrated lift
  `+0.006`, threshold `0.1` harmful-covered `0.087`). Conclusion: keep wins in the acquisition
  lane, but treat flat mixed blocks as calibration/noise and do not assume every added block
  improves the ranker.
- **Block-quality audit added to prevent blind accumulation:** extended
  `scripts/phase8_shop_confidence_audit.py` with a block-quality verdict that preserves win/near-win
  metadata and classifies each labeled block as `strong_signal`, `weak_or_mixed`, or
  `calibration_only` from paired rollout confidence metrics. Focused tests pass
  (`python -m pytest -q tests\test_phase8_shop_confidence_audit.py tests\test_shop_candidate_dataset.py`,
  `18 passed`). Real mixed-block audits match the manual reads: `680000` = `weak_or_mixed`,
  `690000` = `weak_or_mixed`, `700000` = `strong_signal`, `710000` = `calibration_only`.
  A quality-filtered sweep that kept the strong `700000` cheap block plus all r8 confirmations,
  while excluding weak/calibration mixed cheap blocks, produced an 83-example pool with denser
  labels (37 positive / 28 negative / 252 ambiguous, positive label rate `0.117` vs `0.071` in
  the 136 all-in pool)
  (`.data/phase8_ranker_sweep_backward_late_quality_filtered_700strong_plus_r8_h1_m8_confidence_advantage_tie_mse.metrics.json`).
  Both encoders still beat the heuristic on average regret, with larger regret lift on the harder
  filtered set: mean `0.135` vs heuristic `0.200` (6/7 wins), attention `0.132` vs `0.200` (6/7
  wins). Attention has the best calibrated read here (`+0.012` calibrated lift, harmful-covered
  `0.027`), but the dataset is smaller and less stable than the 122 all-in read. Use this as an
  acquisition/training hygiene gate, not yet as a deployment rule.
- **Fifth mixed block confirms win-heavy is not automatically signal-heavy:** offset `720000`
  captured 18 ante-8 snapshots from 9 qualifying trajectories, with wins kept: 12 records from
  6 wins and 6 records from 3 near-wins
  (`.data/phase8_backward_shops_solver_720000_32_late_mixed.jsonl`). Cheap `r=4`, h1 labels were
  win-heavy but flat: heuristic within `0.10` on 15/18 states, mean best-vs-heuristic advantage
  `+0.065`, mean LCB `+0.011`, and practical high-confidence best-vs-heuristic rate `0.056`
  (`.data/phase8_backward_shops_solver_720000_32_late_mixed_r4_h1_m8_meta.jsonl`). The new block
  audit classified it as `weak_or_mixed`
  (`.data/phase8_backward_shops_solver_720000_32_late_mixed_r4_h1_m8_meta.confidence_audit.json`).
  The expanded strict selector found one near-win `open_pack` over `end_shop` candidate, but r8
  confirmation made the exact proposal ambiguous, 0 positive / 0 negative / 1 ambiguous, with
  solver LCB only `+0.012`
  (`.data/phase8_backward_late_154_deepen_select_balanced_terminal_m010_lcb005_pr075_sem080_r8_h1_m8.jsonl`).
  No full 154-state sweep was run because the block gate and r8 audit do not warrant spending that
  compute yet. This reinforces the current rule: keep wins eligible, but require action-level
  signal before treating a winning block as training strength.
- **Quality-filtered checkpoints trained, but live override gate failed:** trained two final
  shop-ranker checkpoints on the quality-gated pool (`84` examples, `44` seeds, strong `700000`
  cheap block plus all r8 confirmations, weak/calibration mixed cheap blocks excluded):
  `.data/phase8_shop_ranker_quality_filtered_attention_v1.pt` and
  `.data/phase8_shop_ranker_quality_filtered_mean_v1.pt`. Training-pool metrics look strong but
  are not deployment evidence: attention final regret `0.087` vs heuristic `0.185`, mean final
  regret `0.077` vs heuristic `0.185`. Probed both checkpoints on fresh offset `730000`, ante-8
  only, conservative baseline-margin gate `0.25`, one ranker action per run. Attention produced
  3 `open_pack` overrides
  (`.data/phase8_ranker_override_attention_qf_v1_730000_32_ante8_m025.jsonl`); mean produced 3
  `open_pack` overrides
  (`.data/phase8_ranker_override_mean_qf_v1_730000_32_ante8_m025.jsonl`). R8 confirmation rejected
  both as deployment candidates: exact proposal audits were `0` positive / `0` negative / `3`
  ambiguous for attention and `0` positive / `0` negative / `3` ambiguous for mean, with no
  high-confidence best-vs-heuristic states. Enhanced `phase8_deepening_confirmation_audit.py` to
  report ranker margins for override-capture records; the ambiguous attention proposals had mean
  ranker baseline margin `0.354`, and mean proposals had `0.397`, so current model margins are not
  calibrated enough for live use. Therefore no live A/B was run. The checkpoints are useful
  proposal/acquisition models, not shop overrides.
- **2026-06-07 action-family separation (buy/card) is the first confirmable override family:** audited
  the override-capture/r8 pipeline (no join bug; CRN pairing valid) and found the 730000 "100%
  open_pack" read was an artifact. A held-out, no-rollout scoring sweep of the attention QF checkpoint
  (offset `740000`, 24 seeds, antes 2-8, gate 0) produced 819 disagreements: 69% open_pack but also
  200 buy/card proposals, and buy/card carried *higher* mean baseline margin (`0.254`) than open_pack
  (`0.181`). open_pack is also nearly unconfirmable at r8 (pack RNG -> wide CIs). Ran a buy/card-only
  override capture on fresh seeds (offset `750000`, 48 seeds, antes 4-6, gate `0.25`, 60 records),
  took the 16 most-confident (margin `0.42`-`0.70`), and r8-confirmed each proposed card vs the
  recomputed heuristic, CRN-paired, **rolled to terminal** via the focus probe
  (`.data/phase8_ranker_override_attn_buycard_750000_top16_r8focus_term.jsonl`). Exact proposal audit:
  **5 positive / 4 negative / 7 ambiguous** (2 of the ambiguous are true ties with identical CRN
  outcomes). The 5 positives are real and large (mean advantage `+0.6`..`+1.6`, LCB `+0.17`..`+0.35`,
  seeds 0750001/0750014/0750023/0750034/0750035). BUT calibration is still broken: positive vs
  ambiguous ranker margins are indistinguishable (`0.530` vs `0.523`), several high-margin proposals
  are strongly negative (LCB to `-1.08`), and the single highest-margin proposal (`0.705`) was a wash.
  Mean realized advantage across all 16 most-confident card-buys is only `+0.065` (within noise), so a
  self-gated override would still net ~neutral. Clean anti-pattern: overriding `end_shop` with a
  card-buy was negative 2/2 (the heuristic is right to stop). Saved the 16-record deployment-
  disagreement block (`.data/phase8_deploy_disagree_block_buycard_750000_r8term.jsonl`) and the 5
  positives (`..._POS5.jsonl`). Verdict: action-family separation yields the project's first
  confirmable positive override labels, but the blocker is now precisely characterized as
  **uncalibrated confidence** (the model learns *which actions can be good*, not *whether they are good
  in this state*), which is a data-scale problem, not a join/labeling bug.
- **2026-06-08 SHOP DEFINITIVELY CLOSED via on-policy value-leaf (the distribution-shift fix did not
  rescue it):** the June-3 conclusion was that prior value heads went flat on deployment shop states
  because they trained off-policy (basic_strategy/bootstrap). Tested the fix head-on: captured 384
  runs FROM THE SOLVER ITSELF (`solver_shop_basic_play_bot`, held-out seeds 2,000,001+, shop-audit
  off; `scripts/phase8_onpolicy_value.py` -> `.data/onpolicy_solver_caps_384.jsonl`, capture winrate
  68/384 = 17.7%), and trained an ATTENTION value net on-policy
  (`.data/value_onpolicy_attn_v1.pt`): val win AUC `0.708`, val ante_corr `0.393` -- real
  whole-state signal. But the smoking-gun check (neural ante_value std on 80 real solver shop states)
  came back **0.053 -- FLATTER than the prior off-policy failure (~0.073)**. So on-policy capture
  did NOT make the value discriminate shop states. This FALSIFIES distribution-shift as the fixable
  cause: the real reason is that eventual outcome barely depends on shop-choice differences (shop
  selection is already near-optimal; cross-candidate variance within one shop is even smaller than
  the 0.053 cross-state std, and the override-ranker independently found only ~1/30 reproducible shop
  disagreements). The winrate A/B was correctly SKIPPED (a flat leaf cannot move the search argmax).
  Conclusion: every shop-value form is now closed -- standalone V(state), rollout, TD(lambda),
  action-ranker override, AND on-policy value-leaf. The trained value net is NOT wasted: AUC 0.708
  means it discriminates winning vs losing whole-run states -- a PLAY/META-level signal, not a
  shop-micro signal. NEXT: stop all shop work; test the only remaining levers (endgame PLAY depth,
  META decisions: skips/tags/vouchers/packs), the work deferred since 2026-06-03.
- **2026-06-08 build-construction localized + first real winrate lever + big efficiency win:**
  - **Out-test (build-ceiling vs play):** for ante-8 losses, replayed each to the failing blind,
    forked the sim, and grafted every affordable offered joker. **42/57 (73.7%) of ante-8 losses
    had >=1 affordable, offered joker that clears the failing blind** (mean ~5 outs each); only ~26%
    were RNG-dead. The bot already plays near-optimally (best-play reaches median 74% of the ante-8
    wall; Violet Vessel only 32%). Conclusion: losses are **build CONSTRUCTION** (reaches the wall
    one affordable-available joker short), not RNG and not play. Scripts: `endgame_play_audit.py`,
    `endgame_out_test.py`.
  - **Economy lever (FIRST confirmed winrate gain):** economy-by-ante audit showed winners diverge
    on money from ante 3 and hold the $25 interest cap far more (0.79 vs 0.62). Added an env-gated
    `BALATRO_ECON_W` interest-discipline multiplier in `shop_search._shop_money_value` (default 1.0
    = inert). A/B (seeds 1..100 then 1..200): `ECON_W=1.5` replicated **+2..+4 winrate, +0.1 mean
    ante, +8 runs to ante 8**, both seed sets, all metrics concordant; `2.0` regresses (over-saves).
    The econ-joker de-saturation term (`BALATRO_ECON_INVEST_W`, mirrors scaling-invest) did NOT
    stack (combo ~= interest-alone). Small but real first lever (~19.5% -> ~21.5%).
  - **Archetype planner Phase 0 (oracle gate):** the archetype system already exists
    (`solver/archetypes.py`: 4 archetypes + fit-score + `ArchetypeAwareLeaf`); the live bot never
    commits (`archetype=None`). Oracle (best-of baseline+4 archetypes per seed, deployed basic-play)
    = 33.3% vs baseline 27.1% on 48 seeds (**+6.2% ceiling, +0.46 mean ante**). But every standalone
    archetype underperforms baseline (12-15%) -> wrong commit hurts; flush is the only strong one.
    So a planner needs a CONSERVATIVE selector (commit margin + hysteresis). Design in
    `ARCHETYPE_PLANNER_PLAN.md`; the neural's right niche is the 4-way archetype SELECTION (learnable),
    not per-card value. **Phase 1 A/B** (`phase8_archetype_planner_ab.py`, live selector via
    `SolverPolicy.archetype` per-decision, 100 seeds, rust on): winrate-NEUTRAL -- aggressive
    `flush_t1` (commit on 1 joker) HURTS (17 vs 22, the wrong-commit penalty); conservative
    `flush_t2`/`general_t2` neutral (23 vs 22). Committing only after the build already owns >=2
    archetype jokers is too late to capture the +6% ceiling, but earlier commit over-commits and
    hurts -- a commit-timing precision problem the joker-count selector can't solve. Would need
    deck-suit-concentration / early flush detection (deferred); modest ceiling makes ROI uncertain.
  - **"Try neural better" (clear-capacity):** reframed the failed win-value leaf to a LOCAL
    clear-capacity target (per-blind build->cleared). It learns build strength (held-out clear-AUC
    0.90; graft gate 0.567 -> 0.698, correct direction) -- a real non-flat neural signal. But
    deployed as a shop leaf it HURTS winrate (5.8% -> 3.3% with weight): it rates overall build
    strength, not which candidate, so it injects noise into the near-optimal shop argmax. Neural can
    MEASURE build strength, not IMPROVE near-optimal shop decisions. Scripts: `phase8_clearcap_train.py`,
    `value_buildgate.py`, `phase8_clearcap_ab.py`.
  - **EFFICIENCY: flipped Rust best-play default ON** (`hand_evaluator._RUST_BESTPLAY_ENABLED`).
    It was fixed/validated 2026-06-05 but left default-OFF, so every run not setting the env
    (winrate benches, diagnostics, the deployed bot) paid full Python subset-enumeration (cProfile:
    `evaluate_played_cards` 362K calls, ~56% of CPU in play). Validated winrate-NEUTRAL (39/200
    IDENTICAL, rust ON vs OFF, seeds 1..200) at ~1.5x faster (1424s -> 919s); 180 rust parity unit
    tests + 345 play/sim/solver tests pass with the new default. Profilers added:
    `profile_deployed.py`, `profile_play.py`. Remaining cost after the flip: shop (`reroll_ev` ~20%,
    `shop_leaf` ~17%).

- **2026-06-08 (cont.) two more BEHAVIOR-IDENTICAL efficiency wins (~12-15% faster, deployed bot):**
  Re-profiled the deployed bot after the rust-default-ON + fast-knob shifts (`profile_deployed.py`
  16/8 = 414s CPU baseline). cProfile (tottime + caller attribution) found two expensive REDUNDANT
  work sources; both fixed with zero behavior change (proven, not just A/B-neutral):
  1. **`hand_draw_odds` deck-rebuild memoization** (`search/hand_viability.py`). The exact draw-odds
     DP was already `lru_cache`d, but `DeckModel.from_cards` + `_deck_signature` (the key-build, ~8%
     of CPU: rebuild the 52-card Counter + rank/suit signature) ran on EVERY call (~78K/run on a deep
     seed). The deck multiset is identity-stable across the thousands of leaf/play evals between card
     acquisitions (measured 99.94% id-repeat on the heavy seed, 46 unique decks / 78K calls). Added a
     bounded (`OrderedDict`, max 256) cache keyed on `id(known_deck)` + the 3 flags (Smeared/Four
     Fingers/Shortcut) + draw size, with the deck tuple PINNED in the value + an `is` guard so an
     id-reuse-after-GC collision is impossible. Equivalence-checked 78,103 calls = 0 mismatch.
  2. **best-play tie-break: stop re-evaluating every tied subset in Python** (`rules/hand_evaluator.
     _fast_winner` + `search/rust_bridge.rust_best_play_scores`). With rust-bestplay ON, the batch
     scores all subsets in Rust, but `_fast_winner` then rebuilt the FULL Python `evaluate_played_cards`
     for EVERY subset tied at the top score (to break ties on `(score,chips,mult)`) -- avg **25 tied
     subsets per call**, ~37K Python evals/seed, the single biggest play cost. Added `with_tiebreak`
     to `rust_best_play_scores`: it now also returns Rust `(score,chips,mult)` for the tied-at-top
     subsets (via the already-exposed `evaluate_simple_with_levels`, reusing parsed joker data; cheap,
     tie-set only; consistency-guarded that detail score == batch top). `_fast_winner` picks the
     `(score,chips,mult)` argmax in enumeration order from Rust detail and builds the Python
     HandEvaluation for ONLY the winner, falling back to the full Python tie-break if detail is
     missing. **Divergence vs the old Python tie-break: 0 / 16,449 tied calls across 15 seeds** (Rust
     == Python chips/mult on the safe path), so plays -- hence winrate -- are unchanged by construction.
  3. **`_pool_records` availability-filter memoization** (`search/shop_sampler.py`). reroll-EV
     sampling calls `_pool_records(pool_type, state)` ~5K/run, each re-filtering the static pool
     (~60 records) via `_record_available` (~280K calls). But `_record_available` reads only fixed
     state.modifiers fields (banned/unlocked/pool_flags/voucher/enhancement), so the ~4 pool lists are
     constant per (state, pool_type) within a decision (measured 79% repeat). Memoized into a per-state,
     identity-pinned, decision-scoped bucket via the existing `_state_scoped_cache` (lazy bots import;
     graceful fallback when no scope). Returned list is read-only at every call site. 0 mismatch /
     10,239 calls. reroll_ev CPU 52s -> 46s (-12%).
  - Net: play CPU 198s->~158s (-20%), reroll_ev -12%; total 414s -> ~360s (~12-15% faster, run-to-run
    var ~4%). 608 tests pass (rust bestplay/hand-eval/viability/odds/deck/search/sim/discard/pack/
    basic-strategy/rng-shop/shop-sampler/solver-policy/solver-play).
  - **Dead ends (reverted, documented so they aren't re-tried):** (a) id-caching
    `_joker_is_disabled_for_build` (350K calls) was perf-NEUTRAL -- the id-cache-lookup + lambda
    overhead cancels the cheap metadata iteration it saves (micro-memoizing cheap-but-frequent fns
    doesn't pay; only eliminating EXPENSIVE redundant work does). (b) Reducing the ~5% best-play
    Rust bails: 100% are boss blinds. The House/Tooth/Fish LOOK safe (0% mismatch on 10 seeds) but
    `rust_bridge` comments document a prior deeper audit -- Tooth's $1/card feeds money-scaling jokers
    (Bull/Bootstraps), House/Fish face-down HELD cards interact with held-card jokers -- so they
    diverge with specific jokers my sample missed; left bailed. Adding Psychic to the batch path
    (zero !=5-card) diverged 8.93% (stateful jokers on 5-card plays); reverted. The author's
    `RUST_BLIND_SAFE` exclusions are correct.
  - Remaining cost profile: shop 55% (`shop_leaf_terms` 30% = build-scoring, mostly content-cached
    Rust evals -- near-irreducible without changing sample count / beam knobs, which need an A/B;
    `reroll_ev` 14%), play 44%. Further free wins look exhausted; bigger gains need behavior changes
    (A/B-gated) or Rust-core work.

- **2026-06-08 (cont.) META blind-SKIP lever TESTED -> net-negative (the one "untested" surface):**
  The pivot memory flagged META (skips/tags/vouchers) as genuinely untested. Found the deployed bot
  **never skips blinds** (`_blind_select_action` always SELECTs; only ever considers a boss reroll).
  Confirmed the sim fully models skip -> tag -> effects in `local_runner` (all 24 tags, incl. pack
  tags Buffoon/Charm/Meteor/Standard, money tags Investment/Economy/Top-up, Rare/Negative/Coupon/D6),
  and `SKIP_BLIND` is legal at every Small/Big blind with the tag visible at decision time
  (`current_blind["tag"]`). So the lever is real and testable.
  - **Paired A/B (100 seeds, each run both ways), skip Small on free-value tags (Buffoon/Charm/
    Meteor/Standard/Negative/Investment/Economy/Voucher/Top-up): winrate 21 -> 11 (-10 net, lost 15 /
    gained 5), mean ante 6.36 -> 6.16.** Strong regression.
  - **Why (flip diagnostic on the 20 changed seeds):** NO static tag/ante pattern separates good from
    bad skips -- the *same* tag (Negative, Economy, Charm, Voucher) at the *same* ante helps on some
    seeds and hurts on others. A skip's value depends on the SPECIFIC forgone shop + build state, so a
    static heuristic can't decide it. Mechanism: skipping forgoes a shop (the bot is build-limited ->
    needs shops) + early blind money/interest (economy matters, cf. ECON_W), and tags add RESOURCES
    not FORESIGHT -- and foresight (which joker clears the future wall) is the actual bottleneck (out-
    test). So META-skip would require counterfactual shop evaluation (sample the would-be shop via the
    existing ShopSampler and compare to the tag), i.e. the same whole-run planning lever -- not a
    heuristic. Code reverted (worktree clean); finding documented. Conclusion stands: every static /
    per-decision lever (shop, economy, flush, neural, META-skip) is now explored; the only remaining
    winrate lever is whole-run build FORESIGHT (search/RL over the forward model -- the PLAN.md /
    PHASE8 end-goal), which the ~13-15% sim speedup this session directly makes cheaper.

- **2026-06-09 S0 FORESIGHT GATE RUN -> FAILS. The archetype-commitment lever is empirically capped
  (~26%) AND not learnable from early state -- the decisive go/no-go for the superhuman build,
  settled for ~1.5hr compute instead of a quarter's engine.** S0 (designed via the multi-agent
  path-design workflow) tested the load-bearing assumption of the AlphaZero-style path: *can early
  visible state predict which build basin the run should commit to?* Pipeline:
  `scripts/phase8_archetype_oracle.py` (+rows), `s0_early_state_capture.py`, `s0_foresight_classifier.py`;
  artifacts `.data/s0_oracle_white_200.json`, `.data/s0_early_features_white_200.json`.
  - **Oracle @ 200 held-out white seeds (deployed backend) REPLICATES +6.5%** (baseline 39/200=19.5%
    -> best-of-4-archetypes 52/200=26.0%, +0.54 mean ante) -- the ceiling is real but **flush-
    dominated**: best_archetype = flush 72 / baseline 124 / scaling+high_card+pair ~1 each; the other
    3 archetypes never help and standalone win ~11-12% (wrong commit hurts, confirms D1).
  - **The hypothesized headline signal -- deck-suit-concentration / "early flush detection" (the
    named-but-deferred fix, PROGRESS.md:2459) -- is EMPIRICALLY NULL.** Deck stays perfectly balanced
    through ante 3 (suit_max_smeared_pair mean 0.500, std 0.0007@a2 / 0.0036@a3): the baseline never
    stacks suits, so suit concentration is a CONSEQUENCE of the flush commit, not a PREDICTOR. Only
    joker-availability + hand-levels + money have early variance.
  - **No early signal predicts the flush basin.** "flush helps this seed" held-out AUC = **0.41-0.51
    across logreg AND MLP, ante2 / ante3 / ante2+3** -- at/below chance (machinery validated: AUC 0.65
    on synthetic planted data). 5-way basin top-1 is BELOW predict-always-baseline. A confidence-gated
    conservative selector captures **0% to NEGATIVE** of the +6.5% gap at every threshold.
  - **Root cause (fundamental):** flush viability is set by the WHOLE RUN's RNG-offered pieces (which
    flush jokers/tarots appear antes 4-8), not observable at the ante-2/3 commit. The +6.5% is a
    perfect-HINDSIGHT artifact; at commit time that knowledge does not exist. A reactive policy/value
    head maps state->action, so absent the deciding info in-state it cannot learn it (AUC-at-chance
    confirms). Steps 4 (live smoke) + 5 (red-stake) are moot -- only meaningful if offline passes.
  - **VERDICT: FAIL -> do NOT build the engine on the archetype lever.** Even a perfect forward-search
    oracle of archetype commitment caps at ~26% (not superhuman); the reactive route is at chance. The
    one measured winrate headroom (+6.2% oracle) is closed as a realizable live lever. ~22-26% is
    at/near the white-stake ceiling for this heuristic architecture; every other surface
    (shop/play/economy/META) is near-optimal/closed. (Caveat: ceiling is for the 4 BUILT-IN
    archetypes; finer build taxonomy = the combinatorial surface that already failed B1/B3.)

- **2026-06-09 from-scratch core investigation -> the construction premise is NOT supported (and is
  not cleanly testable without ante-8 RNG). ~22-26% is at/near this architecture's ceiling.**
  After S0-foresight failed, designed a from-scratch play+build core (NEW_CORE_PLAN.md) on the thesis
  that losses are mid-game build-CONSTRUCTION failures the myopic next-boss leaf misprices. Ran the
  plan's cheap gates:
  - **S-pre (200 deployed runs, winners-vs-losers Cohen's d at matched antes):** the win/loss lever
    is overall BUILD POWER, driven by a COMBINATION the bot already partly handles -- economy
    (money d ~+0.5), more leveling (sum_levels d +0.38), avoiding decay jokers (d -0.49), modestly
    more compounders (d only +0.20). Thesis B (economy/leveling/quality) > thesis A (scaling basin).
    Build-capacity growth: winners sustain ~1.55x/ante; losers collapse 1.55x->1.26x late. Scripts:
    s_pre_out_classify.py, s_pre_winloss_capture.py, s_pre_winloss_analyze.py.
  - **Decay-joker penalty A/B (BALATRO_DECAY_W=1.6, paired 128 seeds):** NEUTRAL (+1/128). The bot
    already handles decay (`_joker_late_durability_factor`); the S-pre decay signal was correlational.
    Knob reverted. -> no cheap heuristic knob left; heuristics are mature.
  - **S0 mid-game CONSTRUCTION kill-switch (s0_midgame_construction.py, 120 seeds) -> NEGATIVE,
    contamination-free.** Fork the sim at antes 3/4/5 shops, force every realizable BUY/SWAP (slots
    are full mid-game), roll to terminal, with NULL + REROLL controls. First run was GENERIC (sampler)
    mode -> contaminated (intervention 8.5% ~= reroll 9.3% per-attempt, all wins from the
    action-dependent sampler). **Re-ran in FAITHFUL mode (balatro_seed -> keyed, action-INDEPENDENT
    shops) and partitioned rollouts by `_rng_diverged`.** v1 faithful run was only ~61% clean and on
    that biased subset showed intervention 1.7%/attempt vs reroll 5.6%/attempt. **DEFINITIVE v2 run
    (`s0_midgame_faithful_v2.json`, after the two source-validated RNG fixes): clean fraction jumped to
    ~100% (intervention 1187/1191, reroll 269/271 stayed seed-faithful), and on that contamination-free
    AND unbiased subset intervention wins 5.0%/attempt (59/1187) vs neutral reroll 4.8%/attempt (13/269)
    -- statistically indistinguishable. The 1.7%-vs-5.6% v1 gap was a bias artifact of the ~61%-clean
    subset, not a real effect.** Only 27/102 losses (26.5%) had ANY clean seed-faithful perturbation
    that flips them, and a forced build buy is no better than a blind reroll at finding it; ~73% of
    losses are unrecoverable mid-game. **Verdict: the bot's mid-game build SELECTION is near-optimal;
    the construction premise is REFUTED -- now contamination-free AND unbiased. ~22-26% is robustly the
    ceiling for build selection.** (Key methodology lesson: always run S0/S-pre-style fork tests in
    FAITHFUL mode (balatro_seed) + partition on `_rng_diverged`; generic mode's action-dependent
    sampler contaminates any fork-intervention test, and a partially-clean faithful subset can still be
    biased -- get the clean fraction near 100% before trusting the magnitudes.)
  - **Bottom line:** every lever this session converges -- S0-foresight (archetype unlearnable, +6.5%
    capped), S-pre (mature heuristics, no cheap knob), decay (neutral), S0-construction (REFUTED
    contamination-free; forced builds < neutral rerolls). ~22-26% is robustly at/near the ceiling for
    THIS architecture -- the bot extracts near-optimally per decision; the gap to human ~80% is
    whole-run POLICY quality, addressable only by large-scale RL self-play (+ validated ante-8 RNG),
    a multi-month program with a robustly-discouraging prior. Banked this session: the ~13-15%
    permanent sim speedup + a definitive ceiling map. The v2 intervention~=reroll parity also closes
    the lone faint hint: "reroll-more-when-behind" (BALATRO_REROLL_BEHIND) was A/B-tested and REGRESSED
    (21->14/128, lost 7 / gained 0) -> reverted; the bot's reroll/economy discipline is near-optimal
    too. EVERY testable per-decision lever is now closed/negative; the bot is at its architecture
    ceiling.

- **2026-06-09 RNG-faithfulness foundation: Bug 2 FIXED (ground-truth-validated), the one remaining
  path is now de-risked from "multi-week + fixtures" to "two bounded fixes, one done."** The "RNG
  unvalidated past ante 5" that blocks deep-search/RL is really just shop-reroll prediction failing
  (the bot rerolls late-ante, `seed_faithful_reroll` returns None -> sampler fallback -> ~35% of
  faithful-mode runs diverge at antes 6-8). Read the decompiled Balatro source (`.data/balatro-source/`)
  and pinpointed TWO bounded bugs. **Bug 2 (planet/consumable exhaustion) FIXED:** the Python
  predictor was missing `get_current_pool`'s empty-pool fallback (common_events.lua:2038-2050 -- when
  all entries UNAVAILABLE, Balatro substitutes ONE default item c_pluto/c_strength/c_incantation/
  j_joker instead of resampling-and-crashing). Added `_pool_or_default` in `rng/surfaces.py`; the
  1-item default yields exactly one rng roll (matches `pseudorandom_element` over [default]).
  **Bug 1 (pool flags) ALSO FIXED:** threaded `state.modifiers['pool_flags']` through the predict
  chain (predict_shop_cards->predict_card->_current_pool->joker_pool_for_rarity, all defaulted) and
  gated the only two flagged jokers (Gros Michel / Cavendish on `gros_michel_extinct`, common_events.
  lua:2027-2028); the default-empty case correctly makes Cavendish UNAVAILABLE pre-extinction (the
  27 wrong predictions). **NET (both fixes): faithful-mode shop divergence 35% -> 18% -> 2%; coverage
  65% -> ~98%; 292 RNG/sim tests pass incl. `test_rng_against_bridge`.** The RNG-foundation blocker
  for deep-search/RL is essentially CLOSED. Files: rng/pools.py, rng/surfaces.py, sim/seed_faithful_
  shop.py. Minor non-blocking residual: ~8 VOUCHER keys mis-predicted in the cash-out voucher path
  (predict `_voucher_pool` vs sampler `_record_available` gating disagree; per-slot fallback, NOT run
  divergence). See [[project_sim_correctness_baseline]].

- **2026-06-09 PERMANENT SPEED WIN: deployed winrate bot 8.9% faster, behaviour-PRESERVING (parity-
  proven -> winrate provably unchanged).** Re-profiled both workloads with process_time (ground truth,
  8-day-old memory confirmed stable): WINRATE-RUN bot (solver_shop_basic_play_bot, the constant A/B
  workload) = shop 54.5% [shop_leaf_terms 30.9%/69.5K calls, reroll_ev 13.4%], play 43.9%, sim.step
  only 1.0% (so optimizing "the sim step" itself is a non-lever); DATA-GEN (SolverPolicy) = play 87%
  [python rollout 31.5%, enum 16%, headroom 14%]. Ran a 36-agent workflow (6 subsystem finders +
  per-candidate adversarial vetters) -> 30 candidates (1 ship / 14 prototype / 15 reject). SHIPPED
  (all parity-verified, combined-signature IDENTICAL to baseline at PYTHONHASHSEED=0):
  (1) `shop_leaf_terms` memoized in the decision-cache (pure fn of owned state, keyed on
  `_sample_build_score_cache_key`+baseline); (2) `_play_candidates` decision-cache memo (identity-
  guarded on state + frozen _BlindContext); (3) lazy `_build_profile`/`_run_plan` in
  `_shop_action_value` (the BUY-card branch — every reroll_ev item — never used them); (4) `reroll_ev`
  hoists `_shop_pressure` to compute ONCE per reroll instead of per sampled item (each item builds a
  fresh temp_state that defeats the id-cache); (5) Rust `scorer.rs` per-action `HashSet<usize>` ->
  stack `[bool;16]` mask in `score_play_actions_batch` (the deployed play scorer — biggest single
  contributor ~2.4%); (6) Rust `evaluate.rs` borrow joker editions/metadata on the length-match path
  instead of `.to_vec()` + skip the `scoring_cards` alloc on the no-joker path; (7) `_cached_simulate_buy`
  decision-scoped memo of the BUY transition (each BUY's `simulate_buy` is otherwise run 2-3x/action — two
  penalty calcs + the real expansion; +0.9%, parity-safe — base buy is deterministic, the overstock rng
  refill stays outside the memo). 1373 tests + 101 Rust
  tests pass. **NEGATIVES (measured, not assumed):** `target-cpu=native` REFUTED — 0% speed AND breaks
  parity (float-reduction reorder flips knife-edge seeds; Balatro scoring is INTEGER chips*mult so
  there's no float SIMD to gain; the win came from removing an *allocation*, not raw compute). Two
  data-gen rollout micro-opts were each PARITY-SAFE but NEUTRAL (~0%) -> reverted: DeckModel-per-
  iteration dedupe, and reusing the best-play score in _should_rollout_discard (score==_score_action
  confirmed on 202/202 probed states). FINDING: the data-gen Python rollout (31.5%) has no cheap
  Python-side win — its cost is already Rust-compute-bound (evaluate/score), which the scorer.rs +
  evaluate.rs changes above DO accelerate. A per-Card RustCard conversion cache (hoist_held_rustcards)
  was also parity-safe but NEUTRAL (+0.1%): both the deployed bot and data-gen use the BATCHED
  `score_play_actions_batch`, not the per-subset `rust_evaluate_score_and_hand_type` path it targeted,
  so the conversion churn isn't on either hot path -> reverted. Correctly-rejected by the vetters: removing the
  `_payload_from_record` deepcopy (it's the only isolation from the shared lru_cache'd pool). **KEY
  METHODOLOGY:** parity A/Bs MUST fix `PYTHONHASHSEED=0` — knife-edge seeds (e.g. seed 10) flip under
  hash randomization regardless of the change, producing spurious 1-seed mismatches; a behaviour-
  preserving (signature-identical) change has provably-unchanged winrate so NO separate winrate A/B is
  needed. Agent speedup estimates ran ~3-5x optimistic vs measured (18%->3%, 9%->~0%) — always measure.
  New reusable tools: `scripts/bot_parity_speed.py` (trajectory-signature + process_time A/B for the
  deployed bot), `scripts/deployed_bot_timing.py` (winrate-bot phase breakdown), `scripts/datagen_
  parity.py` (SolverPolicy parity). Files: search/shop_search.py, search/shop_sampler.py,
  bots/basic_strategy/play_scoring.py + shop_values.py, botlatro-core/src/search/scorer.rs +
  hand_eval/evaluate.rs. See [[project_datagen_speed]].

## Next Steps

> **Top priority:** improve the general unseeded bot through the neural candidate-ranker/search
> path, not seed-known selectors or more local heuristic tuning. Use the portfolio diagnostic
> only to expose complementarity.

1. Stop treating deep shop labels as a one-winner classification problem. Acceptable-set,
   pairwise, and confidence-aware diagnostics now exist; `confidence_advantage_tie_mse` is wired
   but needs a much larger solver-confirmed comparison set before it can be trusted. The current
   merged 54-candidate-label set shows positive validation lift, but not safe calibration. Judge by
   held-out regret, near-best/acceptable-set accuracy, confidence-gated lift, and both conditional
   and covered-state harmful override rates, not top-1 alone.
2. Use the 64-state capture-only pool for label-design experiments. Prefer the diverse targeted
   subset path (`phase8_select_shop_state_pool.py`) before relabeling, and avoid selecting states
   that fail to produce at least two executable probe actions. Fresh2 is exhausted; continue on
   `.data/phase8_capture_pool_v3_128_fresh3.jsonl`, selecting non-overlapping 16-state slices with
   the marginal balance selector. Relabel subsets through `--input-records`; avoid scaling r8
   blindly because the 16-state r8 gate cost 47.6 minutes and still had split-half agreement
   `0.25`.
3. Keep the ranker as a prior or conservative override unless the uncertainty-aware gate proves
   safe. The current mean/attention models can learn some ordering, but train-side threshold
   calibration does not transfer yet, so confidence calibration is not safe enough to replace the
   shop heuristic.
4. Prioritize deployment-disagreement labels before more generic pool labels: scale
   `phase8_ranker_override_capture.py` on fresh held-out seeds, then solver-confirm the captured
   ranker-vs-baseline actions with enough wall time/horizon to be meaningful. Keep strict SEM-gated
   positive acquisition (`max_sem=0.45` first) for build-forward `buy`/`open_pack` choices, and
   treat `end_shop`/skip-economy proposals as a separate action family until their cheap labels
   stop producing solver-confirmed false positives.
5. Add winning-trajectory backward reanalysis as the next late-game label source: capture shop
   snapshots from ante-8 wins and near-wins, branch late shops across legal alternatives, roll
   forward with paired RNG, label by win/ante margin/economy, then gradually move the branch point
   earlier only after late-shop labels are clean.
6. Use `phase8_shop_confidence_audit.py` after every cheap mixed block before spending r8/sweep
   time. Keep wins eligible, but treat `calibration_only` and weak mixed cheap blocks as
   calibration/holdout unless they produce r8-confirmed positives; do not blindly append them to
   training pools.
7. For trained shop-ranker checkpoints, run override-capture plus r8 exact-proposal audit before
   any live A/B. Only spend A/B compute if the checkpoint produces positive, not merely ambiguous,
   deployment-disagreement labels on fresh seeds.
8. Run paired unseeded holdout A/Bs against `solver_shop_basic_play_bot` and `basic_strategy_bot`
   on contiguous seed ranges only after the r8 override gate passes. Promote nothing until
   held-out winrate or regret improves at comparable compute.
9. Scale only after the above gates: larger candidate datasets, batched inference, then iterative
   search-improvement targets where ranker-guided search creates better labels for the next model.
10. Keep sim/RNG validation as the regression gate. The two deferred minor sim bugs (Drunkard sell,
   Credit Card reroll) can be fixed any time, but they are not blocking the current neural probe.
