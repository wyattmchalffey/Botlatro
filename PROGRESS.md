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

## In Progress

- Phase 5: turn the exact scorer into stronger shop/build planning and survival-aware spending.

## Next Steps

1. Tune replacement thresholds after inspecting bad sells and missed upgrades.
2. Use the shop-audit dataset to design a targeted experiment for weak follow-through after early purchases and rich late-game shop skips.
3. Compare losing deep runs against the winning seed to find shop/scoring bottlenecks after ante 6.
4. Add CSV export to replay analyzer if spreadsheet analysis becomes useful.
