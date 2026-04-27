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
- Added standard-library tests for the foundation.

## In Progress

- Phase 4: continue expanding the scorer from exact audited cases into broader joker/enhancement coverage.

## Next Steps

1. Add a broader local joker rarity table for Baseball Card and rarity-aware decisions.
2. Build deterministic test scenarios for remaining complex jokers and card modifiers.
3. Start using the exact scorer to improve discard/play selection and shop valuation.
4. Run larger score-audit sweeps after each new scorer batch.
